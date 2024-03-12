from flask import Flask, request, redirect, url_for
import stripe

app = Flask (__name__)
stripe.api_key = 'API-KEY'


def create_customer(email, card):
    return stripe.Customer.create (email=email, source=stripe.Token.create (card=card))


def charge_customer(customer_id, amount, currency, description):
    return stripe.Charge.create (customer=customer_id, amount=amount, currency=currency, description=description)


@app.route ('/pay/<email>', methods=['POST'])
def pay(email):
    card = {
        'number': request.form.get ('cardNumber'),
        'exp_month': request.form.get ('expiryDate').split ('/') [0],
        'exp_year': request.form.get ('expiryDate').split ('/') [1],
        'cvc': request.form.get ('cvv'),
        'name': request.form.get ('cardHolderName'),
    }
    customer = create_customer (email, card)
    charge_customer (customer.id, 3000, 'usd', 'Service Fee')
    return redirect (url_for ('index'))
