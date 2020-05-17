"""This is the main package of the project that contains all components and runnable scripts.

The whole project consist of functional component modules:

- **database** - contains components with the purpose of data access
- **feature** - contains components for processing of data (extracting features)
- **similarity** - contains components for calculation of item similarity
- **engine** - contains components representing the search engine of the system
- **app** - contains the runnable application of the whole project

and runnable scripts:

- **Enrich entities** - enrich_entities.py
- **CPV extraction** - extract_cpv_codes_from_contracts.py
- **Subject items extraction** - extract_items_from_contracts.py
- **CPV codes import** - import_CPV_codes.py
- **Run application** - run_app.py

  Typical usage example::

    python run_app.py

"""