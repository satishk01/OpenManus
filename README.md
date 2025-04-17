"# OpenManus" 

steps to insall git

sudo yum install -y git

The EC2 instance should have a role associated to access Bedrock models

when you login to the EC2 instance please configure a default provide

aws configure and only set region to us-east-1 or which ever region you are using.

Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh


git clone openManus
https://github.com/satishk01/OpenManus.git

Create a new virtual environment and activate it:
uv venv --python 3.12
source .venv/bin/activate

install dependencies
uv pip install -r requirements.txt

Browser Automation Tool (Optional)
playwright install

run OpenManus:
python main.py

in the folder config there is a file config.toml by default it is set to claude 3.7 we can switch to claude 3.5 v2 based on our availablity