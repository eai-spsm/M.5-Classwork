# W7 Exercise: 16x2 I2C LCD Display
# Prints two lines, refreshing a counter on the second line.
# I2C runs on the Pi's fixed SDA/SCL pins, not a general GPIO pin.
#
# Wiring:
#   LCD VCC -> 5V
#   LCD GND -> GND
#   LCD SDA -> GPIO2 / physical pin 3 (I2C1 SDA, fixed by hardware)
#   LCD SCL -> GPIO3 / physical pin 5 (I2C1 SCL, fixed by hardware)
#
# One-time Pi setup before this will run:
#   sudo raspi-config          # Interface Options -> I2C -> Enable
#   sudo reboot
#   pip install rpi_lcd        # inside your venv
#
# Run `sudo i2cdetect -y 1` first to confirm the LCD's address
# (commonly 0x27 on the PCF8574 backpack used elsewhere in this repo).

from rpi_lcd import LCD
from time import sleep

lcd = LCD()
counter = 0

try:
    while True:
        lcd.text("W7 LCD Test", 1)
        lcd.text(f"Count: {counter}", 2)
        counter += 1
        sleep(1)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    lcd.clear()
