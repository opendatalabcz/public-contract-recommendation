@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SPHINXOPTS    =
set SPHINXBUILD   = sphinx-build
set PAPER         =
set BUILDDIR      = ../build/sphinx/
set AUTODOCDIR    = api
set AUTODOCBUILD  = sphinx-apidoc
set PROJECT       = recommender
set MODULEDIR     = ../src/recommender

set ALLSPHINXOPTS   = -d %BUILDDIR%/doctrees %SPHINXOPTS% .

#set SOURCEDIR=.
#set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:html
	%SPHINXBUILD% -b html %ALLSPHINXOPTS% %BUILDDIR%/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."


:end
popd
