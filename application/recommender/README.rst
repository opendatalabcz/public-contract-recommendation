====================================
PCRec - public contracts recommender
====================================

This project contains the open-source implementation of recommender system for Czech public procurement.

Description
===========

The project is created within a master thesis "Open-source recommender system for Czech public procurement".
Author of the thesis is Milan Vancl <vanclmil@fit.cvut.cz>.

The whole project consist of functional modules:

- **database** - contains components with the purpose of data access
- **feature** - contains components for processing of data (extracting features)
- **similarity** - contains components for calculation of item similarity
- **engine** - contains components representing the search engine of the system
- **app** - contains the runnable application of the whole project

For detailed specification of each module see Documentation_.

Install
=======

To install the project you can simply run::

    python setup.py install

Tests
=====

After the project is installed, you should be able to run tests::

    python setup.py test

The tests require dependency:

- FastText binary model - change the path to the model in `tests/config.cfg` to provide this dependency (model: `wiki.cs
  <https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cs.zip>`_)

Run application
===============

If all tests passed, you can run the demo application with::

    python ./src/recommender/scripts/run_app.py

The application requires a configuration file like `src/recommender/component/app/web/config.cfg`
where you can specify settings for:

- logging (logging level, log file)
- web app session (secret key)
- database connection (db name, user, password, host, port)
- embedding model (path to the FastText binary model, eg. `wiki.cs
  <https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.cs.zip>`_)

You can leave the config file just in your current directory or
provide the path to the config file with system environment variable: PCREC_CONFIG

Documentation
===================

To build the documentation run::

    python setup.py docs

The documentation is built in: `build/sphinx/html/index.html`

..
  Note
  ====

  This project has been set up using PyScaffold 3.2.3. For details and usage
  information on PyScaffold see https://pyscaffold.org/.
