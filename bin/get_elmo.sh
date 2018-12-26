cd models
mkdir -p elmo && cd elmo
# Get Spanish
mkdir -p es && cd es
wget -nc http://vectors.nlpl.eu/repository/11/145.zip
unzip 145.zip
rm 145.zip
# Get English
mkdir -p en && cd en
wget -nc http://vectors.nlpl.eu/repository/11/144.zip
unzip 144.zip
rm 144.zip
