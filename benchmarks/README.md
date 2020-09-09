Benchmarks
==========

pvlib includes a small number of performance benchmarking tests. These
tests are run using
[airspeed velocity](https://asv.readthedocs.io/en/stable/) (ASV).

The basic structure of the tests and how to run them is described below.
We refer readers to the ASV documentation for more details. The AstroPy
[documentation](https://github.com/astropy/astropy-benchmarks/tree/master)
may also be helpful.

The test configuration is described in [asv.conf.json](asv.conf.json).

The performance tests are located in the [benchmarks](benchmarks) directory.

First, from the same directory as `asv.conf.json`, run the tests in a
command prompt (`$`):

```
$ asv run
```

Note that, unlike pytest, the asv tests require changes to be committed
to git before they can be tested. The ``run`` command takes a positional
argument to describe the range of git commits or branches to be tested.
For example, if your feature branch is named ``feature``, a useful asv
run may be:

```
$ asv run master..feature
```

Next, publish the test results to the archive:

```
$ asv publish
```

Finally, start a http server to view the test results:

```
$ asv preview
```

You might also find it useful to compare results from the command line
instead of through the HTML report. After generating timing results by
benchmarking a series of commits as above, or running individual commits
like this:

```
asv run e42f8d24^!
```

You can then compare the timing results of two commits:

```
$ asv compare 0ff98b62 e42f8d24

All benchmarks:

       before           after         ratio
     [0ff98b62]       [e42f8d24]
     <asv_setup~1>       <asv_setup>
+      3.90±0.6ms         31.3±5ms     8.03  irradiance.Irradiance.time_aoi
       3.12±0.4ms       2.94±0.2ms     0.94  irradiance.Irradiance.time_aoi_projection
          256±9ms         267±10ms     1.05  irradiance.Irradiance.time_dirindex
```

The `ratio` column shows the ratio of `after / before` timings. For this
example, the `aoi` function was slowed down on purpose to demonstrate
the comparison.

Nightly benchmarking
--------------------

The benchmarks are run nightly for new commits to pvlib-python/master.

- Timing results: https://pvlib-benchmarker.github.io/pvlib-benchmarks/
- Information on the process: https://github.com/pvlib-benchmarker/pvlib-benchmarks
