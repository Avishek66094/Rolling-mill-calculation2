import base64

with open("Roll pressure function.PNG", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

print(encoded)

import base64

with open("Roll torque function.PNG", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("ascii")

print(encoded)

