# a simple table-access-protocol interface for python

As you know Gaia will produce loads of data and that these will not be easily available as flat table for the next data releases.

I discovered not many people know that it is possible to do quite complex computations in the Gaia database directly, 
i.e, online on the server side and not on your laptop. 
For example doing a color-magnitude diagram using a crossmatch of Tycho colors and Gaia takes 6 seconds, 
when you know how to do it...

This package provides a module interface to TAP as well as not so basic examples from the Gaia DR1.
