import graphviz


def parse_dependency_block(block):
    lines = block.strip ().split ('\n')
    package = lines [0].split (' ') [0]
    dependencies = [line.strip ().split (' ') [0] for line in lines [1:]]
    return package, dependencies


def add_edges(graph, package, dependencies):
    for dep in dependencies:
        graph.edge (package, dep)


def main(dependency_text):
    blocks = dependency_text.strip ().split ('\n\n')
    graph = graphviz.Digraph ('G', filename='dependency_graph.gv')

    for block in blocks:
        package, dependencies = parse_dependency_block (block)
        add_edges (graph, package, dependencies)

    graph.render (view=True)


dependency_text = """
celery==5.4.0rc1
├── billiard [required: >=4.2.0,<5.0, installed: 4.2.0]
├── click [required: >=8.1.2,<9.0, installed: 8.1.7]
│   └── colorama [required: Any, installed: 0.4.6]
├── click-didyoumean [required: >=0.3.0, installed: 0.3.0]
│   └── click [required: >=7, installed: 8.1.7]
│       └── colorama [required: Any, installed: 0.4.6]
├── click-plugins [required: >=1.1.1, installed: 1.1.1]
│   └── click [required: >=4.0, installed: 8.1.7]
│       └── colorama [required: Any, installed: 0.4.6]
├── click-repl [required: >=0.2.0, installed: 0.3.0]
│   ├── click [required: >=7.0, installed: 8.1.7]
│   │   └── colorama [required: Any, installed: 0.4.6]
│   └── prompt-toolkit [required: >=3.0.36, installed: 3.0.43]
│       └── wcwidth [required: Any, installed: 0.2.13]
├── kombu [required: >=5.3.4,<6.0, installed: 5.3.5]
│   ├── amqp [required: >=5.1.1,<6.0.0, installed: 5.2.0]
│   │   └── vine [required: >=5.0.0,<6.0.0, installed: 5.1.0]
│   ├── typing_extensions [required: Any, installed: 4.10.0]
│   └── vine [required: Any, installed: 5.1.0]
├── python-dateutil [required: >=2.8.2, installed: 2.9.0.post0]
│   └── six [required: >=1.5, installed: 1.16.0]
├── tzdata [required: >=2022.7, installed: 2024.1]
└── vine [required: >=5.1.0,<6.0, installed: 5.1.0]
DateTime==5.4
├── pytz [required: Any, installed: 2024.1]
└── zope.interface [required: Any, installed: 6.2]
    └── setuptools [required: Any, installed: 69.1.1]
Flask-Limiter==3.5.1
├── Flask [required: >=2, installed: 3.0.2]
│   ├── blinker [required: >=1.6.2, installed: 1.7.0]
│   ├── click [required: >=8.1.3, installed: 8.1.7]
│   │   └── colorama [required: Any, installed: 0.4.6]
│   ├── importlib_metadata [required: >=3.6.0, installed: 7.0.2]
│   │   └── zipp [required: >=0.5, installed: 3.17.0]
│   ├── itsdangerous [required: >=2.1.2, installed: 2.1.2]
│   ├── Jinja2 [required: >=3.1.2, installed: 3.1.3]
│   │   └── MarkupSafe [required: >=2.0, installed: 2.1.5]
│   └── Werkzeug [required: >=3.0.0, installed: 3.0.1]
│       └── MarkupSafe [required: >=2.1.1, installed: 2.1.5]
├── limits [required: >=2.8, installed: 3.10.0]
│   ├── Deprecated [required: >=1.2, installed: 1.2.14]
│   │   └── wrapt [required: >=1.10,<2, installed: 1.16.0]
│   ├── importlib_resources [required: >=1.3, installed: 6.1.3]
│   │   └── zipp [required: >=3.1.0, installed: 3.17.0]
│   ├── packaging [required: >=21,<24, installed: 23.2]
│   └── typing_extensions [required: Any, installed: 4.10.0]
├── ordered-set [required: >4,<5, installed: 4.1.0]
├── rich [required: >=12,<14, installed: 13.7.1]
│   ├── markdown-it-py [required: >=2.2.0, installed: 3.0.0]
│   │   └── mdurl [required: ~=0.1, installed: 0.1.2]
│   └── Pygments [required: >=2.13.0,<3.0.0, installed: 2.17.2]
└── typing_extensions [required: >=4, installed: 4.10.0]
Flask-Login==0.6.3
├── Flask [required: >=1.0.4, installed: 3.0.2]
│   ├── blinker [required: >=1.6.2, installed: 1.7.0]
│   ├── click [required: >=8.1.3, installed: 8.1.7]
│   │   └── colorama [required: Any, installed: 0.4.6]
│   ├── importlib_metadata [required: >=3.6.0, installed: 7.0.2]
│   │   └── zipp [required: >=0.5, installed: 3.17.0]
│   ├── itsdangerous [required: >=2.1.2, installed: 2.1.2]
│   ├── Jinja2 [required: >=3.1.2, installed: 3.1.3]
│   │   └── MarkupSafe [required: >=2.0, installed: 2.1.5]
│   └── Werkzeug [required: >=3.0.0, installed: 3.0.1]
│       └── MarkupSafe [required: >=2.1.1, installed: 2.1.5]
└── Werkzeug [required: >=1.0.1, installed: 3.0.1]
    └── MarkupSafe [required: >=2.1.1, installed: 2.1.5]
flask-talisman==1.1.0
matplotlib==3.8.3
├── contourpy [required: >=1.0.1, installed: 1.2.0]
│   └── numpy [required: >=1.20,<2.0, installed: 1.26.4]
├── cycler [required: >=0.10, installed: 0.12.1]
├── fonttools [required: >=4.22.0, installed: 4.49.0]
├── importlib_resources [required: >=3.2.0, installed: 6.1.3]
│   └── zipp [required: >=3.1.0, installed: 3.17.0]
├── kiwisolver [required: >=1.3.1, installed: 1.4.5]
├── numpy [required: >=1.21,<2, installed: 1.26.4]
├── packaging [required: >=20.0, installed: 23.2]
├── pillow [required: >=8, installed: 10.2.0]
├── pyparsing [required: >=2.3.1, installed: 3.1.2]
└── python-dateutil [required: >=2.7, installed: 2.9.0.post0]
    └── six [required: >=1.5, installed: 1.16.0]
pandas-datareader==0.10.0
├── lxml [required: Any, installed: 5.1.0]
├── pandas [required: >=0.23, installed: 2.2.1]
│   ├── numpy [required: >=1.22.4,<2, installed: 1.26.4]
│   ├── python-dateutil [required: >=2.8.2, installed: 2.9.0.post0]
│   │   └── six [required: >=1.5, installed: 1.16.0]
│   ├── pytz [required: >=2020.1, installed: 2024.1]
│   └── tzdata [required: >=2022.7, installed: 2024.1]
└── requests [required: >=2.19.0, installed: 2.31.0]
    ├── certifi [required: >=2017.4.17, installed: 2024.2.2]
    ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
    ├── idna [required: >=2.5,<4, installed: 3.6]
    └── urllib3 [required: >=1.21.1,<3, installed: 2.2.1]
pip==24.0
pipdeptree==2.16.1
plotly==5.19.0
├── packaging [required: Any, installed: 23.2]
└── tenacity [required: >=6.2.0, installed: 8.2.3]
qiskit-aer==0.13.3
├── numpy [required: >=1.16.3, installed: 1.26.4]
├── psutil [required: >=5, installed: 5.9.8]
├── qiskit [required: >=0.45.0, installed: 1.0.2]
│   ├── dill [required: >=0.3, installed: 0.3.8]
│   ├── numpy [required: >=1.17,<2, installed: 1.26.4]
│   ├── python-dateutil [required: >=2.8.0, installed: 2.9.0.post0]
│   │   └── six [required: >=1.5, installed: 1.16.0]
│   ├── rustworkx [required: >=0.14.0, installed: 0.14.1]
│   │   └── numpy [required: >=1.16.0,<2, installed: 1.26.4]
│   ├── scipy [required: >=1.5, installed: 1.11.4]
│   │   └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.4]
│   ├── stevedore [required: >=3.0.0, installed: 5.2.0]
│   │   └── pbr [required: >=2.0.0,!=2.1.0, installed: 6.0.0]
│   ├── symengine [required: >=0.11, installed: 0.11.0]
│   ├── sympy [required: >=1.3, installed: 1.12]
│   │   └── mpmath [required: >=0.19, installed: 1.3.0]
│   └── typing_extensions [required: Any, installed: 4.10.0]
└── scipy [required: >=1.0, installed: 1.11.4]
    └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.4]
qutip==5.0.0b1
├── numpy [required: >=1.22, installed: 1.26.4]
├── packaging [required: Any, installed: 23.2]
└── scipy [required: >=1.8,<1.12, installed: 1.11.4]
    └── numpy [required: >=1.21.6,<1.28.0, installed: 1.26.4]
wheel==0.42.0
yfinance==0.2.37
├── appdirs [required: >=1.4.4, installed: 1.4.4]
├── beautifulsoup4 [required: >=4.11.1, installed: 4.12.3]
│   └── soupsieve [required: >1.2, installed: 2.5]
├── frozendict [required: >=2.3.4, installed: 2.4.0]
├── html5lib [required: >=1.1, installed: 1.1]
│   ├── six [required: >=1.9, installed: 1.16.0]
│   └── webencodings [required: Any, installed: 0.5.1]
├── lxml [required: >=4.9.1, installed: 5.1.0]
├── multitasking [required: >=0.0.7, installed: 0.0.11]
├── numpy [required: >=1.16.5, installed: 1.26.4]
├── pandas [required: >=1.3.0, installed: 2.2.1]
│   ├── numpy [required: >=1.22.4,<2, installed: 1.26.4]
│   ├── python-dateutil [required: >=2.8.2, installed: 2.9.0.post0]
│   │   └── six [required: >=1.5, installed: 1.16.0]
│   ├── pytz [required: >=2020.1, installed: 2024.1]
│   └── tzdata [required: >=2022.7, installed: 2024.1]
├── peewee [required: >=3.16.2, installed: 3.17.1]
├── pytz [required: >=2022.5, installed: 2024.1]
└── requests [required: >=2.31, installed: 2.31.0]
    ├── certifi [required: >=2017.4.17, installed: 2024.2.2]
    ├── charset-normalizer [required: >=2,<4, installed: 3.3.2]
    ├── idna [required: >=2.5,<4, installed: 3.6]
    └── urllib3 [required: >=1.21.1,<3, installed: 2.2.1]
"""
if __name__ == "__main__":
    main (dependency_text)
