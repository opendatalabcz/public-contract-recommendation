from flask_wtf import FlaskForm
from wtforms import StringField


class ContractSearchForm(FlaskForm):

    search = StringField('')

    def get_query(self):
        return {'subject': self.search.data}


class ProfileForm(FlaskForm):
    choices = [('Artist', 'Artist'),
               ('Album', 'Album'),
               ('Publisher', 'Publisher')]
    # select = SelectField('Search for music:', choices=choices)
    search = StringField('')
