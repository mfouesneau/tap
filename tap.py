""" A short package to interogate TAP services """
import requests
import time
import math
try:  # python 3
    from io import BytesIO
    from http.client import HTTPConnection
    from urllib.parse import urlencode
except ImportError:  # python 2
    from StringIO import StringIO as BytesIO
    from httplib import HTTPConnection
    from urllib import urlencode

from xml.dom.minidom import parseString
from lxml import etree
import json
from astropy.table import Table

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = None
    display = None


def _pretty_print_time(t):
    """ Print time with units """
    units = [u"s", u"ms", u'us', "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    if t > 0.0 and t < 1000.0:
        order = min(-int(math.floor(math.log10(t)) // 3), 3)
    elif t >= 1000.0:
        order = 0
    else:
        order = 3
    return "%.3g %s" % (t * scaling[order], units[order])


class TAP_AsyncQuery(object):
    """ Asynchronous Query

    Attributes
    ---------
    host: str
        tap host
    path: str
        path to the service on host
    port: int
        port of the service
    adql_query: str
        query
    session: Session object
        use a given requests.Session to proceed
        esp. useful with authenticated sessions
    """
    def __init__(self, adql_query, host, path, port=80, session=None):
        """ set the query """
        self.adql = adql_query
        self.host = host
        self.port = port
        self.path = path
        self.location = None
        self.jobid = None
        self.response = None
        self.session = session

    def submit(self, silent=False):
        """ Submit the query to the server

        Parameters
        ----------
        silent: bool
            prints some information if not set
        """
        data = {'query': str(self.adql),
                'request': 'doQuery',
                'lang': 'adql',
                'format': 'votable',
                'phase': 'run'}

        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept":       "text/plain"
        }

        # add authentication and other cookies to the header
        try:
            cookies = self.session.cookies
            headers['Cookie'] = ';'.join("{0}={1}".format(k,v) for k,v in cookies.items())
        except:
            pass
        connection = HTTPConnection(self.host, self.port)
        connection.request("POST", self.path, urlencode(data), headers)
        #Status
        self.response = connection.getresponse()
        #Server job location (URL)
        self.location = self.response.getheader("location")
        #Jobid
        self.jobid = self.location[self.location.rfind('/') + 1:]
        connection.close()

        if not silent:
            print("Query Status: " + str(self.response.status),
                  "Reason: " + str(self.response.reason))
            print("Location: " + self.location)
            print("Job id: " + self.jobid)

    @classmethod
    def recall_query(cls, host, path, port, jobid):
        """ Connect to a remote job
        Parameters
        ----------
        host: str
            tap host
        path: str
            path to the service on host
        port: int
            port of the service
        jobid: str
            job identifier

        Returns
        -------
        query: TAP_AsyncQuery
            asynchrone query object
        """
        location = '{1:s}/{2:s}/async/{0:s}'.format(jobid, host, path)
        q = cls("", host, path, port)
        q.location = location
        q.jobid = jobid
        return q

    @property
    def status(self):
        """ Check job status on the server """
        headers = {}
        # add authentication and other cookies to the header
        try:
            self.response = self.session.get('http://' + self.location)
            data = self.response.text
        except:
            connection = HTTPConnection(self.host, self.port)
            connection.request("GET", self.path + "/" + self.jobid, headers)
            self.response = connection.getresponse()
            data = self.response.read()
        #XML response: parse it to obtain the current status
        dom = parseString(data)
        phase_element = dom.getElementsByTagName('uws:phase')[0]
        phase_value_element = phase_element.firstChild
        phase = phase_value_element.toxml()
        return phase

    @property
    def finished(self):
        """ Check if job done """
        return self.status == 'COMPLETED'

    def get(self, sleep=0.2, wait=True):
        """
        Get the result or wait until ready

        Parameters
        ----------
        sleep: float
            Delay between status update for a given number of seconds
        wait: bool
            set to wait until result is ready

        Returns
        -------
        table: Astropy.Table
            votable result
        """
        while (not self.finished) & wait:
            time.sleep(sleep)
        if not self.finished:
            return
        #Get results
        try:
            self.response = self.session.get('http://' + self.location + "/results/result")
            self.data = self.response.text
        except:
            connection = HTTPConnection(self.host, self.port)
            connection.request("GET", self.path + "/" + self.jobid + "/results/result")
            self.response = connection.getresponse()
            self.data = self.response.read()
            connection.close()
        try:
            table = Table.read(BytesIO(self.data), format="votable")
            return table
        except TypeError:
            table = Table.read(BytesIO(self.data.encode('utf8')), format="votable")
            return table
        except Exception as e:
            content = parseString(self.response.text)
            text = []
            for k in content.getElementsByTagName('INFO'):
                name, value = k.attributes.values()
                if 'QUERY_STATUS' in value.nodeValue:
                    status = value.nodeValue
                    print(status)
                    text.append(k.firstChild.nodeValue.replace('.', '.\n').replace(':', ':\n'))
            print(e)
            raise RuntimeError('Query error.\n{0}'.format('\n'.join(text)))

    def _repr_markdown_(self):
        try:
            from IPython.display import Markdown

            return Markdown("""*ADQL Query*\n```mysql\n{0}\n```\n* *Status*:   `{1}`, Reason `{2}`\n* *Location*: {3}\n* *Job id*:   `{4}`\n
                            """.format(str(self.adql), str(self.response.status),
                                       str(self.response.reason),
                                       self.location, self.jobid))._repr_markdown_()
        except ImportError:
            pass


class TAP_Service(object):
    """
    Attributes
    ----------
    host: str
        tap host
    path: str
        path to the service on host
    port: int
        port of the service
    adql_query: str
        query
    """
    def __init__(self, host, path, port=80, **kargs):
        self.host = host
        self.port = port
        self.path = path
        self.session = requests.Session()

    @property
    def tap_endpoint(self):
        """ Full path """
        return "http://{s.host:s}{s.path:s}".format(s=self)

    def reacall_query(self, jobid):
        """ Connect to a remote job
        Parameters
        ----------
        jobid: str
            job identifier

        Returns
        -------
        query: TAP_AsyncQuery
            asynchrone query object
        """
        location = '{1:s}{2:s}/async/{0:s}'.format(jobid, self.host, self.path)
        q = TAP_AsyncQuery("", self.host, self.path, self.port, self.session)
        q.location = location
        q.jobid = jobid
        return q

    def login(self, username, password=None):
        """
        Login to the service
        Password is not stored with this object, only the cookie will be

        Parameters
        ----------
        username: string
            username to use with this service
        password: string, optional
            password. If not provided, will prompt for it (not stored later)
        """
        if password is None:
            import getpass
            pw = getpass.getpass()
        r = self.session.post("https://{s.host:s}/tap-server/login".format(s=self),
                              data={'username': username, 'password':pw})
        if not r.ok:
            raise RuntimeError('Authentication failed\n' + str(r))

    def logout(self):
        """
        Logout from the service
        """
        return self.session.post("https://{s.host:s}/tap-server/logout".format(s=self))

    def query(self, adql_query, sync=True):
        """
        Query a TAP service synchronously
        with a given ADQL query

        Parameters
        ----------
        adql_query: str
            query to send

        Returns
        -------
        tab: Astropy.Table
            votable result
        """
        if sync:
            r = self.session.post(self.tap_endpoint + '/sync',
                                  data={'query': str(adql_query),
                                        'request': 'doQuery',
                                        'lang': 'adql',
                                        'format': 'votable',
                                        'phase': 'run'}
                                  )
            try:
                table = Table.read(BytesIO(r.text.encode('utf8')),
                                   format="votable")
                return table
            except:  # help debugging
                self.response = r
                content = parseString(self.response.text)
                text = []
                for k in content.getElementsByTagName('INFO'):
                    name, value = k.attributes.values()
                    if 'QUERY_STATUS' in value.nodeValue:
                        status = value.nodeValue
                        print(status)
                        text.append(k.firstChild.nodeValue.replace('.', '.\n').replace(':', ':\n'))
                raise RuntimeError('Query error.\n{0}'.format('\n'.join(text)))
        else:
            return self.query_async(adql_query)

    def query_async(self, adql_query, submit=True, **kwargs):
        """ Send an async query

         Parameters
        ----------
        adql_query: str
            query to send
        submit: bool
            set to submit the query otherwise
            returns the constructed query that
            can be submitted later.

        Returns
        -------
        query: TAP_AsyncQuery
            Query object
        """
        q = TAP_AsyncQuery(adql_query, self.host,
                           self.path + '/async',
                           port=self.port,
                           session=self.session)
        if submit:
            q.submit(**kwargs)
        return q

    def get_table_list(self):
        """ returns the list of available tables from the service

        ADQL query: SELECT * from TAP_SCHEMA.tables

        Returns
        -------
        tab: Table
            list of the tables with description, size and types
        """
        return self.query("""select * from TAP_SCHEMA.tables where schema_name not like 'tap_schema'""")

    def get_table_info(self, tablename):
        return self.query("Select top 0 * from {0} ".format(tablename))

    def upload_table(self, table_name, path, **kwargs):
        """
        upload a table to your private local space in the archive.

        Parameters
        ----------
        table_name: str
            The name to assign to this table.
        path: str
            The local path of the table to upload

        Returns
        -------
        success: bool
            True if the upload was successful.
        """

        with open(path, "r") as fp:
            url = self.tap_endpoint[:-3] + 'Upload'
            self.response = self.session.post(url,
                                              files=dict(FILE=fp),
                                              data=dict(TABLE_NAME=table_name))
        if not self.response.ok:
            raise RuntimeError("Upload failed")

        return True


class TAPVizieR(TAP_Service):
    """ TAPVizier / CDS TAP service """
    def __init__(self, *args, **kwargs):
        host = 'tapvizier.u-strasbg.fr'
        path = '/TAPVizieR/tap'
        port = 80
        TAP_Service.__init__(self, host, path, port, *args, **kwargs)


class GaiaArchive(TAP_Service):
    def __init__(self, *args, **kwargs):
        host = "gea.esac.esa.int"
        port = 80
        path = "/tap-server/tap"
        TAP_Service.__init__(self, host, path, port, *args, **kwargs)


def resolve(objectName):
        """
        Resolve the object by name using CDS

        Parameters
        ----------
        objectName: str
            Name to resolve

        Returns
        -------
        ra: float
            right ascension
        dec: float
            declination

        Example:
        >> resolve('M31')
        (10.684708329999999, 41.268749999999997)

        Requires the following module: lxml
        """
        host = "cdsweb.u-strasbg.fr"
        port = 80
        path = "/cgi-bin/nph-sesame/-ox?{0}".format(objectName)
        connection = HTTPConnection(host, port)
        connection.request("GET", path)
        response = connection.getresponse()
        xml = response.read()
        try:
            tree = etree.fromstring(xml.encode('utf-8'))
        except:
            tree = etree.fromstring(xml)
        # take the first resolver
        pathRa = tree.xpath('/Sesame/Target/Resolver[1]/jradeg')
        pathDec = tree.xpath('/Sesame/Target/Resolver[1]/jdedeg')
        if len(pathRa) == 0:
            return []
        ra = float(pathRa[0].text)
        dec = float(pathDec[0].text)
        return ra,dec


class QueryStr(object):
    """ A Query string that also shows well in notebook mode"""
    def __init__(self, adql, *args, **kwargs):
        verbose = kwargs.pop('verbose', True)
        self.text = adql
        self._parser = 'https://sqlformat.org/api/v1/format'
        self._pars = {'sql': adql, 'reindent': 0, 'keyword_case': 'upper'}
        self.parse_sql(**kwargs)
        if verbose:
            try:
                display(self)
            except:
                print(self)

    def parse_sql(self, **kwargs):
        self._pars.update(**kwargs)
        res = requests.post(self._parser, self._pars)
        self.text = json.loads(res.text)['result']
        return self

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text

    def _repr_markdown_(self):
        try:
            return Markdown("""*ADQL query*\n```mysql\n{0:s}\n```""".format(self.text))._repr_markdown_()
        except:
            pass


class timeit(object):
    """ Time a block of code of function.
        Works as a context manager or decorator.
    """
    def __init__(self, func=None):
        self.func = func
        self.text = ''

    def __str__(self):
        return "*Execution time*: {0}".format(self.text)

    def _repr_markdown_(self):
        try:
            return Markdown("*Execution time*: {0}".format(self.text))._repr_markdown_()
        except:
            pass

    def __call__(self, *args, **kwargs):
        if self.func is None:
            return
        with timeit():
            result = self.func(*args, **kwargs)
        return result

    @classmethod
    def _pretty_print_time(cls, t):
        units = [u"s", u"ms",u'us',"ns"]
        scaling = [1, 1e3, 1e6, 1e9]
        if t > 0.0 and t < 1000.0:
            order = min(-int(math.floor(math.log10(t)) // 3), 3)
        elif t >= 1000.0:
            order = 0
        else:
            order = 3

        return "%.3g %s" % (t * scaling[order], units[order])

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args, **kwargs):
        self.stop = time.time()
        self.text = self._pretty_print_time(self.stop - self.start)
        display(self)
