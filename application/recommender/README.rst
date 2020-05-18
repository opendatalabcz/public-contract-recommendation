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

Database initialization
=======================

    Disclaimer: this is the most complex and time consuming part of application management
    (for instance, time estimation of collecting and processing one year data is in the order of a few weeks).

The whole project is built upon specific data so it is necessary to initialize a database.

By default a Postgres database is used (there is already an implementation of Postgres data access layer used by the application).

Tha main part of data used in this project is managed by parallel project public-contracts_ so to create and initialize a Postgres database follow its instructions.

.. _public-contracts: https://github.com/opendatalabcz/public-contracts

To finalize the initialization run following scripts to process all data:

- enrich_entities.py_ - Enriches submitter information
- import_CPV_codes.py_ - Imports CPV codes from CSV
- extract_cpv_codes_from_contracts.py_ - Extracts CPV codes from contracts
- extract_items_from_contracts.py_ - Extracts subject items from contracts

.. _enrich_entities.py: src/recommender/scripts/enrich_entities.py
.. _import_CPV_codes.py: src/recommender/scripts/import_CPV_codes.py
.. _extract_cpv_codes_from_contracts.py: src/recommender/scripts/extract_cpv_codes_from_contracts.py
.. _extract_items_from_contracts.py: src/recommender/scripts/extract_items_from_contracts.py

Run application
===============

If all tests passed and the database is created, you can run the demo application with::

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
