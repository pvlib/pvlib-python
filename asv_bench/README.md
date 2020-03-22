Benchmarks
==========

pvlib includes a small number of performance benchmarking tests. These
tests are run using
[airspeed velocity](https://asv.readthedocs.io/en/stable/) (ASV).

The basic structure of the tests and how to run them is described below.
We refer readers to the ASV documentation for more details.

The test configuration is described in
[``asv.conf.json``](asv.conf.json).

The performance tests are located in the [benchmarks](benchmarks) directory.

First, from this directory, run the tests:

```
asv run
```

Note that, unlike pytest, the asv tests require changes to be committed
to git before they can be tested. The ``run`` command takes a positional
argument to describe the range of git commits or branches to be tested.
For example, if your feature branch is named ``feature``, a useful asv
run may be:

```
asv run master..feature
```

Next, publish the test results to the archive:

```
asv publish
```

Finally, start a http server to view the test results:

```
asv preview
```

Other useful commands
---------------------

The argument passed to `asv run` has the same syntax as `git log` and is
therefore pretty powerful.  For instance, you can run a specific tag with
`asv run v0.6.0^!`. 

If a benchmark function is failing and you don't know why, the `-e` option
will display error messages:

```
asv run -e
```