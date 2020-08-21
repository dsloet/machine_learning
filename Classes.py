# Classes.py is a module containing some Classes.

class Car():
    """Model a car."""

    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year


        self.fuel_capacity = 15
        self.fuel_level = 0

    def fill_tank(self):
        """Fill gas tank to capacity"""
        self.fuel_level = self.fuel_capacity
        print("Fuel tank is full")

    def drive(self):
        """Simulate driving"""
        print("The car is moving")

    def update_fuel_level(self, new_level):
        """Update the fuel level"""
        if new_level <= self.fuel_capacity:
            self.fuel_level = new_level
        else:
            print("The tank can't hold that much!")

    def add_fuel(self, amount):
        if(self.fuel_level + amount <= self.fuel_capacity):
            self.fuel_level += amount
            print("Added fuel. New fuel level = %f" % self.fuel_level)
        else:
            print("The tank can't hold that much!")


class ElectricCar(Car):
    """Simple EV car"""

    def __init__(self, make, model, year):
        """Initialise an EC"""
        super().__init__(make, model, year)

        # Attributes specific to EC
        # Battery cap in kWh
        self.battery_size = 70
        # Charge level in %
        self.charge_level = 0

    def charge(self):
        """Fully charge the battery"""

        self.charge_level = 100
        print("The battery is fully charged.")

    # Override fill_tank
    def fill_tank(self):
        """Display error message"""
        print("This car has no fuel tank!")


class Battery():
    """A battery for an EC"""

    def __init__(self, size=70):
        """Initialize battery attributes"""
        # Capacity in kWh, charge level in %
        self.size = size
        self.charge_level = 0

    def get_range(self):
        """Retrun the battery's range."""

        if self.size == 70:
            return 240
        elif self.size == 85:
            return 270

class ElectricCar2(Car):
    """Simple EV car"""

    def __init__(self, make, model, year):
        """Initialise an EC"""
        super().__init__(make, model, year)

        # Attributes specific to EC
        # Add Battery() instance as an attribute
        self.battery = Battery()

    def charge(self):
        """Fully charge the battery"""

        self.charge_level = 100
        print("The battery is fully charged.")

    # Override fill_tank
    def fill_tank(self):
        """Display error message"""
        print("This car has no fuel tank!")

