import os
import sys
# noinspection PyPackageRequirements
from sphinx.ext.autodoc import ClassLevelDocumenter
# noinspection PyPackageRequirements
from sphinx.ext.autodoc import InstanceAttributeDocumenter
# noinspection PyPackageRequirements
from sphinx.util.typing import stringify as stringify_type_hint

# Add library root to path.
sys.path.insert(0, os.path.abspath('../..'))

# ----------------------------------------------------------------------------
# Monkey patch in order to not show default values of instance attributes.
# The default values are not taken from constructor, thus they are wrong.
# ----------------------------------------------------------------------------
SUPPRESS = object()


def i_add_directive_header(self, sig):
    ClassLevelDocumenter.add_directive_header(self, sig)
    source_name = self.get_sourcename()
    if not self.options.annotation:
        if not self._datadescriptor:
            # obtain annotation for this attribute
            annotations = getattr(self.parent, '__annotations__', {})
            if annotations and self.objpath[-1] in annotations:
                objrepr = stringify_type_hint(
                    annotations.get(self.objpath[-1])
                )
                self.add_line('   :type: ' + objrepr, source_name)
            else:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if self.analyzer and key in self.analyzer.annotations:
                    v = self.analyzer.annotations[key]
                    if v == "_np.ndarray":
                        v = "ndarray"
                    elif v == "_typing.Sequence[str]":
                        v = "list of str"
                    else:
                        v = v.replace("_typing.", "")
                        v = v.replace("_np.", "")
                    self.add_line(f"   :type: {v}", source_name)

            # ## Remove (= None).
            # try:
            #     obj_repr = object_description(self.object)
            #     self.add_line('   :value: ' + obj_repr, source_name)
            # except ValueError:
            #     pass
    elif self.options.annotation is SUPPRESS:
        pass
    else:
        self.add_line('   :annotation: %s' % self.options.annotation,
                      source_name)


InstanceAttributeDocumenter.add_directive_header = i_add_directive_header
# ----------------------------------------------------------------------------
# End of monkey patch.
# ----------------------------------------------------------------------------

# Project information.
project = 'bio_rtd'
# noinspection PyShadowingBuiltins
copyright = '2020, University of Natural Resources and Life Sciences (BOKU),' \
            ' Vienna'
author = 'Jure Sencar'
version = '0.7.1'

# General configuration.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'bokeh.sphinxext.bokeh_plot',
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

autodoc_member_order = 'bysource'


# Allow referencing sections by title.
autosectionlabel_maxdepth = 2

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
#     'numpy': ('https://docs.scipy.org/doc/numpy/', None)
# }


# Link to templates. In our case we want to remove "Edit on Github" link.
templates_path = ['_templates']

exclude_patterns = []

pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
