# W1 Exercise: Terminal ATM
# No GPIO/hardware here — plain Python exercise: login, check balance,
# deposit, withdraw with banknote breakdown.

balance = 5000
pin = "1234"
notes = [1000, 500, 100, 50, 20]

entered_pin = input("Enter PIN: ")
if entered_pin != pin:
    print("Wrong PIN.")
    exit()

while True:
    print("\n1) Balance  2) Deposit  3) Withdraw  4) Exit")
    choice = input("Choose: ")

    if choice == "1":
        print(f"Balance: {balance} THB")

    elif choice == "2":
        amount = int(input("Deposit amount: "))
        balance += amount
        print(f"New balance: {balance} THB")

    elif choice == "3":
        amount = int(input("Withdraw amount: "))
        if amount > balance:
            print("Not enough balance.")
            continue

        remaining = amount
        print(f"Dispensing {amount} THB:")
        for note in notes:
            count = remaining // note
            remaining = remaining % note
            if count > 0:
                print(f"  {count} x {note} THB")

        if remaining != 0:
            print("Cannot dispense exact amount with available notes.")
            continue

        balance -= amount
        print(f"New balance: {balance} THB")

    elif choice == "4":
        print("Goodbye.")
        break

    else:
        print("Invalid choice.")
