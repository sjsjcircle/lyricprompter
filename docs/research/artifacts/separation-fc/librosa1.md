# Librosa 1.0
This release is intended to stabilize the API from the long-running 0.x series.

A full changelog is available at https://librosa.org/doc//dev/changelog.html

A [blog post](https://librosa.org/blog/posts/1.0/)  on the 1.0 release gives a high level overview of the changes.

## What's Changed
* fix: correct typo in core/convert.py line:1658 by @afadel151 in https://github.com/librosa/librosa/pull/1932
* Test overhaul by @bmcfee in https://github.com/librosa/librosa/pull/1925
* Peak picking by dynamic programming by @bmcfee in https://github.com/librosa/librosa/pull/1927
* Trying a new environment cache setup in CI by @bmcfee in https://github.com/librosa/librosa/pull/1936
* Variable-rate and non-iterative phase vocoding by @bmcfee in https://github.com/librosa/librosa/pull/1931
* Include scipy-stubs in the test dependencies - v1.0.0 rebase by @jorenham in https://github.com/librosa/librosa/pull/1946
* vscale option to specshow by @bmcfee in https://github.com/librosa/librosa/pull/1938
* Remove default quantization for note_to_hz by @bmcfee in https://github.com/librosa/librosa/pull/1947
* Freq convert nan by @bmcfee in https://github.com/librosa/librosa/pull/1948
* Colorbar helpers by @bmcfee in https://github.com/librosa/librosa/pull/1952
* Two-slope norm for inferred diverging colormap by @bmcfee in https://github.com/librosa/librosa/pull/1954
* Warn on perceptual weighting underflow by @bmcfee in https://github.com/librosa/librosa/pull/1955
* Warping path to step grid by @bmcfee in https://github.com/librosa/librosa/pull/1958
* Fix subsegment example by @bmcfee in https://github.com/librosa/librosa/pull/1959
* SPEC07 compliant rng parameters by @bmcfee in https://github.com/librosa/librosa/pull/1960
* Upload doc build preview artifacts by @bmcfee in https://github.com/librosa/librosa/pull/1962
* Expire deprecations for 1.0 release by @bmcfee in https://github.com/librosa/librosa/pull/1963
* Signal mixing helpers by @bmcfee in https://github.com/librosa/librosa/pull/1964
* oct3 display modes by @bmcfee in https://github.com/librosa/librosa/pull/1967
* Inverted waveshow by @bmcfee in https://github.com/librosa/librosa/pull/1971
* Wavebars by @bmcfee in https://github.com/librosa/librosa/pull/1973
* Refine .gitignore for improved accuracy and coverage by @MLsound in https://github.com/librosa/librosa/pull/1969
* Fixing figures clumped up at the end of page by @haydnllh in https://github.com/librosa/librosa/pull/1979
* Fix types in beat tracking code by @emilazy in https://github.com/librosa/librosa/pull/1977
* Fix #1970: Prevent AdaptiveWaveplot adaptor from being garbage collected by @Daniel-Haas-B in https://github.com/librosa/librosa/pull/1992
* Add Metrogram Transform by @danferns in https://github.com/librosa/librosa/pull/1986
* Update type annotations by @bmcfee in https://github.com/librosa/librosa/pull/1980
* Migrate from SciPy spmatrix to sparray by @Daniel-Haas-B in https://github.com/librosa/librosa/pull/1994
* Python 3.14 support by @bmcfee in https://github.com/librosa/librosa/pull/1999
* feat: implement hybrid_tempogram (#1991) by @Gopesh111 in https://github.com/librosa/librosa/pull/2004
* Document specshow frequency axis scaling by @cookesan in https://github.com/librosa/librosa/pull/2006
* Surface contributing guide in docs site, link build instructions by @petrakuhnle in https://github.com/librosa/librosa/pull/2009
* Multiplot by @bmcfee in https://github.com/librosa/librosa/pull/2012
* Spec0 2026.06 by @bmcfee in https://github.com/librosa/librosa/pull/2018
* display.wavef0 (was: melodyplot) by @bmcfee in https://github.com/librosa/librosa/pull/1976
* Yet another Viterbi acceleration PR by @bmcfee in https://github.com/librosa/librosa/pull/2007
* Updates for mypy 2.1 by @bmcfee in https://github.com/librosa/librosa/pull/2025
* Fix tonnetz display label ordering by @bmcfee in https://github.com/librosa/librosa/pull/2026
* Modern linting by @bmcfee in https://github.com/librosa/librosa/pull/2027
* Dynamic vectorize by @bmcfee in https://github.com/librosa/librosa/pull/2028
* Even faster import by @bmcfee in https://github.com/librosa/librosa/pull/2029
* Negative offset load by @bmcfee in https://github.com/librosa/librosa/pull/2030
* Loadx by @bmcfee in https://github.com/librosa/librosa/pull/2031
* more helpful failure mode in loadx by @bmcfee in https://github.com/librosa/librosa/pull/2032
* Modernize types in `librosa._typing` by @jorenham in https://github.com/librosa/librosa/pull/2035
* modernizing type annotations for the future by @bmcfee in https://github.com/librosa/librosa/pull/2036
* more type annotation modernization by @bmcfee in https://github.com/librosa/librosa/pull/2038
* Fill in missing type annotations by @jorenham in https://github.com/librosa/librosa/pull/2037
* Display Highlighter by @bmcfee in https://github.com/librosa/librosa/pull/2034
* cmap_cyclic= in specshow by @bmcfee in https://github.com/librosa/librosa/pull/2041
* avoid turning scalars into 0-d arrays in `core.{convert,spectrum}` and `util.utils` by @jorenham in https://github.com/librosa/librosa/pull/2039
* Waveplot styling by @bmcfee in https://github.com/librosa/librosa/pull/2044
* improved `dtype_r2c` and `dtype_c2r` typing in `util.utils` by @jorenham in https://github.com/librosa/librosa/pull/2043
* Doc check modernizing by @bmcfee in https://github.com/librosa/librosa/pull/2047
* Contributing and agents update by @bmcfee in https://github.com/librosa/librosa/pull/2017
* reformat `**/__init__.pyi` re-exports by @jorenham in https://github.com/librosa/librosa/pull/2049
* Include str dtype in `ndarray` return type annotations by @jorenham in https://github.com/librosa/librosa/pull/2050
* Multichannel db by @bmcfee in https://github.com/librosa/librosa/pull/2051
* Fix decorator callable types in `librosa.util.decorators` by @jorenham in https://github.com/librosa/librosa/pull/2053
* Improve `ndarray` return types by @jorenham in https://github.com/librosa/librosa/pull/2052
* Support matplotlib 3.11 in tests by @bmcfee in https://github.com/librosa/librosa/pull/2054
* Stream resampling by @bmcfee in https://github.com/librosa/librosa/pull/2055
* massive simplification of legend_for_axes, fixes #2057 by @bmcfee in https://github.com/librosa/librosa/pull/2059
* Refactor frequency formatters by @bmcfee in https://github.com/librosa/librosa/pull/2060
* Updating actions by @bmcfee in https://github.com/librosa/librosa/pull/2065
* Standardize n_fft parameter docstrings by @brookcs3 in https://github.com/librosa/librosa/pull/2068
* Precise dtype type annotations in `librosa.beat` by @jorenham in https://github.com/librosa/librosa/pull/2069
* Infer n_bins automatically in cqt/vqt by @bmcfee in https://github.com/librosa/librosa/pull/2066
* Add out= parameter to to_mono for in-place downmixing by @ejwong24 in https://github.com/librosa/librosa/pull/2058
* Fix RQA single-axis inputs by @suhashollakc in https://github.com/librosa/librosa/pull/2063
* BoundedNorm for specshow on boolean data by @bmcfee in https://github.com/librosa/librosa/pull/2072
* Skip redundant np.abs copy in piptrack for non-negative real input by @imrulhuda in https://github.com/librosa/librosa/pull/2080
* Tutorial restructure and general documentation overhaul by @bmcfee in https://github.com/librosa/librosa/pull/1968
* 1.0.0rc0 release by @bmcfee in https://github.com/librosa/librosa/pull/2082
* revising nnls initialization by @bmcfee in https://github.com/librosa/librosa/pull/2085
* Validate audio input in feature.rms (closes #2086) by @Kayvan-Zahiri in https://github.com/librosa/librosa/pull/2087
* 1.0 final by @bmcfee in https://github.com/librosa/librosa/pull/2084

## New Contributors
* @afadel151 made their first contribution in https://github.com/librosa/librosa/pull/1932
* @jorenham made their first contribution in https://github.com/librosa/librosa/pull/1946
* @MLsound made their first contribution in https://github.com/librosa/librosa/pull/1969
* @haydnllh made their first contribution in https://github.com/librosa/librosa/pull/1979
* @emilazy made their first contribution in https://github.com/librosa/librosa/pull/1977
* @Daniel-Haas-B made their first contribution in https://github.com/librosa/librosa/pull/1992
* @danferns made their first contribution in https://github.com/librosa/librosa/pull/1986
* @Gopesh111 made their first contribution in https://github.com/librosa/librosa/pull/2004
* @cookesan made their first contribution in https://github.com/librosa/librosa/pull/2006
* @petrakuhnle made their first contribution in https://github.com/librosa/librosa/pull/2009
* @brookcs3 made their first contribution in https://github.com/librosa/librosa/pull/2068
* @ejwong24 made their first contribution in https://github.com/librosa/librosa/pull/2058
* @suhashollakc made their first contribution in https://github.com/librosa/librosa/pull/2063
* @imrulhuda made their first contribution in https://github.com/librosa/librosa/pull/2080
* @Kayvan-Zahiri made their first contribution in https://github.com/librosa/librosa/pull/2087

**Full Changelog**: https://github.com/librosa/librosa/compare/0.11.0...1.0.0
