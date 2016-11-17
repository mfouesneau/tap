# a simple table-access-protocol interface for python

As you know Gaia will produce loads of data and that these will not be easily available as flat table for the next data releases.

I discovered not many people know that it is possible to do quite complex computations in the Gaia database directly, 
i.e, online on the server side and not on your laptop. 
For example doing a color-magnitude diagram using a crossmatch of Tycho colors and Gaia takes 6 seconds, 
when you know how to do it...

This package provides a module interface to TAP as well as not so basic examples from the Gaia DR1.


## Content

**Some common code to send ADQL Queries to TAP services and notebook polishing**

* `GaiaArchive` is a shortcut from `TAP_service` to the interface with the Gaia Archive
* `TAPVizieR`   is a shortcut from `TAP_service` to the interface with TAP service of VizieR (CDS).
* `resolve`     interfaces CDS/Sesame name resolver to get positions of known objects by their names.
* `QueryStr`    is a polished string that parses an SQL syntax to make it look nicer (for notebook and console)
* `timeit`      a context manager/decorator that reports execution time (for notebook and console)


## Quick Start: How to query TAP with this package?

Let's start by checking that we can access the data by requesting the first 5 sources in TGAS.

**Synchronous mode**

Get the service and submit the query. The result will be downloaded automatically.

```python
from tap import GaiaArchive
gaia = GaiaArchive()
result = gaia.query("select top 5 * from gaiadr1.gaia_source")
```

**Asynchronous mode**

Get the service and submit the query. The result will be downloaded automatically.

```python
from tap import GaiaArchive
gaia = GaiaArchive()
query = gaia.querya_sync("select top 5 * from gaiadr1.gaia_source")
```

use `get` to obtain the data later on
```python
result = query.get()
```

**authentication**

use the `login` and `logout` methods to authenticate with your account.
Any `async` query will be stored into your account.

```python
from tap import GaiaArchive
gaia = GaiaArchive()
gaia.login("username", "password")
# or
gaia.login("username")
```
Note: Password will be requested (if not provided) but never stored.

**Recall a previous job (asynchronous only)**

```python
from tap import GaiaArchive
gaia = GaiaArchive()
query = gaia.recall_query("job identifier")
result = query.get()
```

## What is TAP ?

The entry point is a TAP (Table Access Protocol) server.

TAP provides two operation modes, **Synchronous** and **Asynchronous**.

* **Synchronous**: the response to the request will be generated as soon as the request received by the server.

* **Asynchronous**: the server will start a job that will execute the request.
  The first response to the request is the required information (a link) to obtain the job status. 
  Once the job is finished, the results can be retrieved.


## Gaia Archive TAP service
Gaia Archive TAP server provides two access modes: *public* and *authenticated*

* **Public**: this is the standard TAP access. 
  A user can execute ADQL queries and upload tables to be used in a query 'on-the-fly' (these tables will be removed once the query is executed). 
  The results are available to any other user and they will remain in the server for a limited space of time.
  
* **Authenticated**: some functionalities are restricted to authenticated users only.
  The results are saved in a private user space and they will remain in the server for ever (they can be removed by the user). ADQL queries and results are saved in a user private area.

* *Cross-match operations*: a catalogue cross-match operation can be executed. 
  Cross-match operations results are saved in a user private area.
  

## What is ADQL?

ADQL = Astronomical Data Query Language

ADQL has been developed based on `SQL92` and supports a subset of the SQL grammar with extensions to support generic and astronomy specific operations.

In other words, ADQL is a SQL-like searching language improved with geometrical functions.

for more information see the IVOA documentation http://www.ivoa.net/documents/latest/ADQL.html

* examples of SQL minimal queries

```mysql
SELECT *
FROM "gaiadr1.tgas_source"
```

```mysql
SELECT top 1000 ra, dec, phot_g_mean_mag AS mag  
FROM "gaiadr1.gaia_source"
ORDER BY mag
```
