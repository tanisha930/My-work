print("🧾 Welcome to Invoice Generator\n")

# Step 1: Take number of items
num_items = int(input("How many items to bill? "))

# Step 2: Create empty invoice list
invoice = []

# Step 3: Take item details
for i in range(num_items):
    print(f"\nItem {i+1}:")
    name = input("Item name: ")
    price = float(input("Item price: "))
    quantity = int(input("Quantity: "))
    total = price * quantity
    invoice.append({"name": name, "price": price, "quantity": quantity, "total": total})

# Step 4: Print invoice summary
print("\n🔸 Invoice Summary:")
subtotal = 0
for item in invoice:
    print(f"{item['name']} - {item['quantity']} x {item['price']} = {item['total']}")
    subtotal += item['total']

# Step 5: Add tax and total
tax = subtotal * 0.05
grand_total = subtotal + tax
print(f"\nSubtotal: {subtotal}")
print(f"Tax (5%): {tax}")
print(f"Grand Total: {grand_total}")