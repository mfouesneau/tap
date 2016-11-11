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
from astropy.table import Table


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
    """
    def __init__(self, adql_query, host, path, port=80):
        """ set the query """
        self.adql = adql_query
        self.host = host
        self.port = port
        self.path = path
        self.location = None
        self.jobid = None
        self.response = None

    def submit(self, silent=False):
        """ Submit the query to the server

        Parameters
        ----------
        silent: bool
            prints some information if not set
        """
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept":       "text/plain"
        }

        data = {'query': str(self.adql),
                'request': 'doQuery',
                'lang': 'adql',
                'format': 'votable',
                'phase': 'run'}

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

    @property
    def status(self):
        """ Check job status on the server """
        connection = HTTPConnection(self.host, self.port)
        connection.request("GET", self.path + "/" + self.jobid)
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
        connection = HTTPConnection(self.host, self.port)
        connection.request("GET", self.path + "/" + self.jobid + "/results/result")
        self.response = connection.getresponse()
        data = self.response.read()
        table = Table.read(BytesIO(data), format="votable")
        connection.close()
        return table

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

    @property
    def tap_endpoint(self):
        """ Full path """
        return "http://{s.host:s}{s.path:s}".format(s=self)

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
            r = requests.post(self.tap_endpoint + '/sync',
                              data={'query': str(adql_query),
                                    'request': 'doQuery',
                                    'lang': 'adql',
                                    'format': 'votable',
                                    'phase': 'run'}
                              )
            table = Table.read(BytesIO(r.text.encode('utf8')),
                               format="votable")
            return table
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
                           port=self.port)
        if submit:
            q.submit(**kwargs)
        return q


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
