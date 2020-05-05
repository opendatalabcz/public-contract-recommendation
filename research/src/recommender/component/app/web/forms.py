from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField


class BaseSearchForm(FlaskForm):
    search = StringField('')

    def get_query(self):
        return {'subject': self.search.data}


class ContractSearchForm(FlaskForm):
    subject = TextAreaField('Předmět zakázky', render_kw={'class': 'full_line', 'id': 'item_enumeration'})
    address = StringField('Lokalita (adresa zadavatele)', render_kw={'class': 'full_line'})
    entity_subject = TextAreaField('Předmět podnikání zadavatele', render_kw={'class': 'full_line', 'id': 'item_enumeration'})

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
    locality = StringField('Lokalita', render_kw={'class': 'full_line'})
    interest_items = TextAreaField('Zájmové položky', render_kw={'class': 'full_line', 'id': 'item_enumeration'})

    def init_with_profile(self, user_profile):
        self.locality.data = user_profile.locality.address
        self.interest_items.data = '\n'.join([item.description for item in user_profile.interest_items])


class LoginForm(FlaskForm):
    icologin = StringField('Podle IČO')
    idlogin = StringField('Podle ID')
