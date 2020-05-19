import numpy

from flask import session, render_template, request, flash, redirect
from flask_login import login_user, login_required, logout_user, current_user

from recommender.component.app.web.forms import ContractSearchForm, ProfileForm, BaseSearchForm, LoginForm


def init_app(app):
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        # Here we use a class of some kind to represent and validate our
        # client-side form data. For example, WTForms is a library that will
        # handle this for us, and we use a custom LoginForm to validate.
        loginform = LoginForm()
        if loginform.validate_on_submit():
            # Login and validate the user.
            # user should be an instance of your `User` class
            data = loginform.icologin.data + loginform.idlogin.data
            if data != '':
                user = app.load_user_from_loginform(loginform)
                if user:
                    login_user(user)
                    return redirect('/')
            flash('Vyplňte jednu z možností přihlášení!')
        return render_template('login.html', form=loginform)

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect('/')

    @app.route('/', methods=['GET', 'POST'])
    def index():
        searchform = BaseSearchForm()
        if searchform.validate_on_submit():
            data = searchform.search.data
            if not data == '':
                return search_results(searchform)
        if current_user.is_authenticated:
            user_profile = current_user.user_profile
            contracts = app.recommend(user_profile)
            locality_contracts = app.recommend(user_profile, {'locality'}, 3)
            subject_contracts = app.recommend(user_profile, {'subject'}, 3)
            entity_subject_contracts = app.recommend(user_profile, {'entity_subject'}, 3)
            return render_template('index.html', contracts=contracts, searchform=searchform, user_profile=user_profile,
                                   locality_contracts=locality_contracts, subject_contracts=subject_contracts,
                                   entity_subject_contracts=entity_subject_contracts)
        else:
            contracts = app.get_contracts(numpy.random.randint(0, 1000, size=10).tolist())
            return render_template('index.html', contracts=contracts, searchform=searchform)

    @app.route('/search', methods=['GET', 'POST'])
    def search():
        searchform = ContractSearchForm()
        if searchform.validate_on_submit():
            data = searchform.subject.data + searchform.address.data + searchform.entity_subject.data
            if not data == '':
                return search_results(searchform)
        contracts = app.get_contracts(numpy.random.randint(0, 1000, size=10).tolist())
        return render_template('search.html', contracts=contracts, searchform=searchform)

    @app.route('/contract/<int:contract_id>', methods=['GET'])
    def contract(contract_id):
        contracts = app.get_contracts([contract_id])
        if len(contracts) == 0:
            flash('Zakázka id {} neexistuje!'.format(contract_id))
        contract = contracts[0]
        if current_user.is_authenticated:
            app.update_user_profile(contract)
        return render_template('contract_detail.html', contract=contract)

    @app.route('/results')
    def search_results(searchform):
        contracts = app.search(searchform)
        return render_template('search_result.html', contracts=contracts)

    @app.route('/profil', methods=['GET', 'POST'])
    def profil():
        form = ProfileForm(request.form)
        user_profile = current_user.user_profile
        if request.method == 'POST':
            app.save_profile(form)
            flash('Změny uloženy!')
            return redirect('/profil')
        form.init_with_profile(user_profile)
        return render_template('profile.html', form=form, user_profile=user_profile)
