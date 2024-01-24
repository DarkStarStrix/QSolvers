import stripe

# generate a key from stripe.com and paste it here account details to get the key
stripe.api_key = 'API-KEY'
user_email = 'EMAIL'
# Create a customer
customer = stripe.Customer.create (
    email=user_email,
    source=stripe.Token.create (
        card={
            'number': '4242424242424242',
            'exp_month': 12,
            'exp_year': 2021,
            'cvc': '123',
        },
    ),
)


@app.route ('/pay/<email>', methods=['POST'])
def pay(email):
    # Get the payment form data
    card_number = request.form.get ('cardNumber')
    expiry_date = request.form.get ('expiryDate')
    cvv = request.form.get ('cvv')
    card_holder_name = request.form.get ('cardHolderName')

    # Split the expiry date into month and year
    month, year = expiry_date.split ('/')

    # Create a new Stripe customer
    customer = stripe.Customer.create (
        email=email,
        source=stripe.Token.create (
            card={
                'number': card_number,
                'exp_month': month,
                'exp_year': year,
                'cvc': cvv,
                'name': card_holder_name,
            },
        ),
    )

    # Charge the customer
    stripe.Charge.create (
        customer=customer.id,
        amount=3000,  # Amount in cents
        currency='usd',
        description='Service Fee',
    )

    # If the payment is successful, redirect the user to the index page
    return redirect (url_for ('index'))
