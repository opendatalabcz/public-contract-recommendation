import re
import requests

from recommender.component.base import Component


class MapyParser:
    """Parser of the Mapy.cz API response."""
    def __init__(self):
        self._xpattern = re.compile(r'x="([\d]+\.[\d]*)"')
        self._ypattern = re.compile(r'y="([\d]+\.[\d]*)"')

    def parse(self, xml_string):
        """Parses x and y points from xml data of the Mapy.cz API response.

        Args:
            xml_string (str): xml string to be parsed

        Returns:
            tuple: y,x point
        """
        if 'item' in xml_string:
            x = self._xpattern.search(xml_string).group(1)
            y = self._ypattern.search(xml_string).group(1)
            return y, x
        return None


class APIGeocoder(Component):
    """Interface for the Mapy.cz API geocoding service."""
    def __init__(self, api_url='https://api.mapy.cz/geocode?query={}', response_parser=MapyParser(), **kwargs):
        """
        Args:
            api_url (str): formating string containing API url and query parameter
            response_parser (Object): parser of the API response data
        """
        super().__init__(**kwargs)
        self._api_url = api_url
        self._parser = response_parser

    def gps_for_address(self, address):
        """Gets gps coordinations for an address

        Provides the API request and parsing.

        Args:
            address (str): address of free text format

        Returns:
            tuple: x,y tuple representing gps point
        """
        self.print('Getting GPS for address: {}'.format(address), 'debug')
        url = self._api_url.format(address)
        response = requests.get(url)
        if response.status_code == 200:
            gps = self._parser.parse(response.content.decode('utf-8'))
            self.print('Found GPS: {}'.format(gps), 'debug')
            return gps
        self.print('Erorr {} for url {}'.format(response.status_code, url), 'error')
        return None
