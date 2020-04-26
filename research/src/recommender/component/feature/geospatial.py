import re
import requests

from recommender.component.base import Component


class MapyParser:

    def __init__(self):
        self._xpattern = re.compile(r'x="([\d]+\.[\d]*)"')
        self._ypattern = re.compile(r'y="([\d]+\.[\d]*)"')

    def parse(self, xml_string):
        if 'item' in xml_string:
            x = self._xpattern.search(xml_string).group(1)
            y = self._ypattern.search(xml_string).group(1)
            return y, x
        return None


class APIGeocoder(Component):

    def __init__(self, api_url='https://api.mapy.cz/geocode?query={}', response_parser=MapyParser(), **kwargs):
        super().__init__(**kwargs)
        self._api_url = api_url
        self._parser = response_parser

    def gps_for_address(self, address):
        self.print('Getting GPS for address: {}'.format(address), 'debug')
        url = self._api_url.format(address)
        response = requests.get(url)
        if response.status_code == 200:
            gps = self._parser.parse(response.content.decode('utf-8'))
            self.print('Found GPS: {}'.format(gps), 'debug')
            return gps
        self.print('Erorr {} for url {}'.format(response.status_code, url), 'error')
        return None
