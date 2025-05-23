# Currency Converter Project

def convert_currency(amount, from_currency, to_currency):
    exchange_rates = {
        ("USD", "BDT"): 110.0,
        ("BDT", "USD"): 0.0091,
        ("EUR", "BDT"): 117.0,
        ("BDT", "EUR"): 0.0085,
        ("USD", "EUR"): 0.91,
        ("EUR", "USD"): 1.1
    }

    key = (from_currency, to_currency)
    if key in exchange_rates:
        return amount * exchange_rates[key]
    else:
        return None


def main():
    print("Welcome to the Currency Converter!")
    
    try:
        amount = float(input("Enter amount to convert: "))
        from_currency = input("Enter FROM currency (e.g., USD, BDT, EUR): ").upper()
        to_currency = input("Enter TO currency (e.g., USD, BDT, EUR): ").upper()

        result = convert_currency(amount, from_currency, to_currency)

        if result is not None:
            print(f"{amount} {from_currency} = {result:.2f} {to_currency}")
        else:
            print("Conversion rate not found for this currency pair.")
    except ValueError:
        print("Please enter a valid number for amount.")


if _name_ == "_main_":
    main()