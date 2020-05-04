from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField


class BaseSearchForm(FlaskForm):

    search = StringField('')

    def get_query(self):
        return {'subject': self.search.data}


class ContractSearchForm(FlaskForm):

    subject = TextAreaField('Předmět zakázky')
    address = StringField('Lokalita (adresa zadavatele)')
    entity_subject = TextAreaField('Předmět podnikání zadavatele')

    def get_query(self):
        query = {}
        if self.subject.data != '':
            query['subject'] = self.subject.data
        if self.address.data != '':
            query['locality'] = self.address.data
        if self.entity_subject.data != '':
            query['entity_subject'] = self.entity_subject.data
        return query


class ProfileForm(FlaskForm):
    choices = [('Artist', 'Artist'),
               ('Album', 'Album'),
               ('Publisher', 'Publisher')]
    # select = SelectField('Search for music:', choices=choices)
    search = StringField('')
