if [ ! -d "./CircuitsVis" ]; then
    git clone https://github.com/tomasruizt/CircuitsVis
fi
uv pip install -e ./CircuitsVis/python