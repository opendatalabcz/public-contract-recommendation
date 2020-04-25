import re
import requests


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


class Geocoder:

    def __init__(self, api_url='https://api.mapy.cz/geocode?query={}'):
        self._api_url = api_url
        self._parser = MapyParser()

    def gps_for_address(self, address):
        response = requests.get(self._api_url.format(address))
        if response.status_code == 200:
            gps = self._parser.parse(response.content.decode('utf-8'))
            return gps
