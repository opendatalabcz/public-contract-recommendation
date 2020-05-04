from flask import session, render_template, request, flash

from recommender.component.app.web.forms import ContractSearchForm, ProfileForm, BaseSearchForm


def init_app(app):
    @app.route('/set/')
    def set():
        session['key'] = 'value'
        return 'ok'

    @app.route('/get/')
    def get():
        return session.get('key', 'not set')

    @app.route('/', methods=['GET', 'POST'])
    def index():
        searchform = BaseSearchForm()
        if searchform.validate_on_submit():
            data = searchform.search.data
            if not data == '':
                return search_results(searchform)
        contracts = app.get_contracts(((1,9,10,12, 839, 70, 849, 6),))
        return render_template('index.html', contracts=contracts, searchform=searchform)

    @app.route('/search', methods=['GET', 'POST'])
    def search():
        searchform = ContractSearchForm()
        if searchform.validate_on_submit():
            data = searchform.subject.data + searchform.address.data + searchform.entity_subject.data
            if not data == '':
                return search_results(searchform)
        contracts = app.get_contracts(((1,9,10,12, 839, 70, 849, 6),))
        return render_template('search.html', contracts=contracts, searchform=searchform)

    @app.route('/contract/<int:contract_id>', methods=['GET'])
    def contract(contract_id):
        contracts = app.get_contracts([contract_id])
        if len(contracts) == 0:
            flash('Zakázka id {} neexistuje!'.format(contract_id))
        contract = contracts[0]
        return render_template('contract_detail.html', contract=contract)

    @app.route('/results')
    def search_results(searchform):
        contracts = app.search(searchform)
        return render_template('search_result.html', contracts=contracts)

    @app.route('/profil', methods=['GET', 'POST'])
    def profil():
        form = ProfileForm(request.form)
        if request.method == 'POST':
            flash('Změny uloženy!')
            return render_template('profile.html', form=form)
        return render_template('profile.html', form=form)

    @app.route('/test', methods=['GET'])
    def test():
        return render_template('test.html')
