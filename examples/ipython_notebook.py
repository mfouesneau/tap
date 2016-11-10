"""
Some tools for the notebooks
"""
from IPython.display import display, Markdown
try:
    from nbconvert.filters.markdown import markdown2latex, markdown2html
except ImportError:
    from IPython.nbconvert.filters.markdown import markdown2latex, markdown2html
from IPython.display import DisplayObject
import time as _time
import sys


class Caption(Markdown):
    """ Make a caption to associate with figures """
    def __init__(self, s, center=False, **kwargs):
        Markdown.__init__(self, s, **kwargs)
        self._center = center

    def _repr_html_(self):
        txt = markdown2html(self.data)
        if self._center:
            return '<center>{0}</center>'.format(txt)
        else:
            return '{0}'.format(txt)

    def _repr_latex_(self):
        txt = markdown2latex(self.data)
        if self._center:
            return '\\begin{center}\n' + txt + '\n\\end{center}'
        else:
            return txt

    def display(self):
        display(self)

    def __str__(self):
        return self._repr_latex_()


class Matrix(object):
    """ Make a caption to associate with figures """
    def __init__(self,s, fmt='%0.4g'):
        self.s = s
        self._fmt = fmt

    def _repr_(self):
        text = r"""\begin{bmatrix}"""

        t = []
        for k in self.s:
            t.append( ' & '.join([self._fmt % v for v in k] ) + r'\\' )
        text += ''.join(t)
        text += r"""\end{bmatrix}"""
        return Markdown(text)

    def _repr_latex_(self):
        text = r"""\begin{bmatrix}"""

        t = []
        for k in self.s:
            t.append( ' & '.join([self._fmt % v for v in k] ) + r'\\' )
        text += ''.join(t)
        text += r"""\end{bmatrix}"""
        return text

    def __str__(self):
        return self._repr_latex_()

    def display(self):
        display(self)


def disp_markdown(*args):
    return display(Markdown(*args))


def load_latex_macros():
    return disp_markdown(open('notebook_macros').read())


def add_input_toggle():
    from IPython.display import HTML, display

    r = HTML('''
    <script>

    $( document ).ready(function () {
        IPython.CodeCell.options_default['cm_config']['lineWrapping'] = true;
        IPython.notebook.get_selected_cell()

        IPython.toolbar.add_buttons_group([
                {
                    'label'   : 'toggle all input cells',
                    'icon'    : 'fa-eye-slash',
                    'callback': function(){ $('div.input').slideToggle(); }
                }
            ]);
    });
    </script>
    ''')
    display(r)
    return r


def add_citation_button():
    from IPython.display import HTML, display
    r = HTML("""
    <script>

    function insert_citn() {
        // Build paragraphs of cell type and count

        var entry_box = $('<input type="text"/>');
        var body = $('<div><p> Enter the Bibtex reference to insert </p><form>').append(entry_box)
                    .append('</form></div>');

        // Show a modal dialog with the stats
        IPython.dialog.modal({
            notebook: IPython.notebook,
            keyboard_manager: IPython.notebook.keyboard_manager,
            title: "Bibtex reference insertion",
            body: body,
            open: function() {
                // Submit on pressing enter
                var that = $(this);
                that.find('form').submit(function () {
                    that.find('.btn-primary').first().click();
                    return false;
                });
                entry_box.focus();
            },
            buttons : {
                "Cancel" : {},
                "Insert" : {
                    "class" : "btn-primary",
                    "click" : function() {
                        // Retrieve the selected citation, add to metadata,
                        var citation = entry_box.val();
                        // if (!citation) {return;}
                        var citn_html = '<cite data-cite="' + citation + '">' + citation + '</cite>';
                        var cell = IPython.notebook.get_selected_cell();
                        cell.code_mirror.replaceSelection(citn_html);
                    }
                }
            }
        });
    };

    $( document ).ready(function () {

        IPython.toolbar.add_buttons_group([
                {
                    'label'   : 'insert bibtex reference in markdown',
                    'icon'    : 'fa-graduation-cap', // http://fontawesome.io/icons/
                    'callback': insert_citn,
                }
            ]);
    });

    </script>
    <style>
    cite {
        font-style: normal;
        color: #45749e;
    }
    </style>
    """)
    display(r)
    return r


class PDF(object):
    def __init__(self,url):
        self.url = url

    def _repr_html_(self):
        return '<iframe src=%s></iframe>' % self.url

    def _repr_latex_(self):
        return r'\begin{center} \adjustimage{max size={0.9\linewidth}{0.9\paperheight}}{%s}\end{center}' % self.url


class Table(DisplayObject):
    VDOTS = object()

    def __init__(self, data, headings=None, formats=None, caption=None,
                 label=None, position='h', subtables=1):
        """
        A HTML/LaTeX IPython DisplayObject Table

        `data` should be a 2 dimensional array, indexed by row then column,
        with an optional extra row `headings`.

        A 'row' (i.e., an element of `data`) may also be
        :py:const:`Table.VDOTS`, which produces vertical dots in all columns.

        `formats` may be a string, whose format method will be used for every
        cell; a function, called for every cell; or a mixed array of strings
        and functions which is zipped with each row.
        Headings are not formatted.

        `caption` and `label` add the relevant LaTeX markup, and will go in
        the first row of the HTML copy. `label` will have ``tab:`` prepended
        to it.

        If `subtables` is greater than 1, the table will be split into
        `subtables` parts of approximately equal length, and laid out side
        by side.
        """

        if len(data) == 0:
            raise ValueError("data is empty")

        if label is None != caption is None:
            raise ValueError("specify neither or both of label & caption")

        self.columns = len(data[0])
        if self.columns == 0:
            raise ValueError("no columns")

        if headings and len(headings) != self.columns:
            raise ValueError("bad headings length")

        if isinstance(formats, str):
            formats = [formats.format] * self.columns
        elif callable(formats):
            formats = [formats] * self.columns
        elif formats:
            if len(formats) != self.columns:
                raise ValueError("bad formats length")

            def maybe_string_format(f):
                if isinstance(f, str):
                    return f.format
                else:
                    assert callable(f)
                    return f

            formats = list(map(maybe_string_format, formats))
        else:
            formats = [self._default_format] * self.columns

        for i, row in enumerate(data):
            if row is not self.VDOTS and len(row) != self.columns:
                raise ValueError("bad row length", i)

        self.headings = headings
        self.data = data
        self.formats = formats
        self.caption = caption
        self.label = label
        self.position = position
        self.subtables = subtables

    @staticmethod
    def _default_format(what):
        if isinstance(what, float):
            return "{0:.5f}".format(what)
        else:
            return str(what)

    def _format_rows(self):
        for row in self.data:
            if row is self.VDOTS:
                yield self.VDOTS
            else:
                yield (f(x) for f, x in zip(self.formats, row))

    def _subtables_split(self):
        assert self.subtables > 1

        rows = list(self._format_rows())
        nominal_height = len(rows) // self.subtables
        remainder = len(rows) % self.subtables

        heights = [nominal_height] * self.subtables
        for i in range(remainder):
            heights[i] += 1

        slices = []
        acc = 0
        for l in heights:
            slices.append((acc, acc + l))
            acc += l
        assert slices[-1][1] == len(rows)

        subtables = [rows[a:b] for a, b in slices]
        return subtables

    def _repr_latex_(self):
        strings = []

        strings.append(r"""
        \begin{table}[""" + self.position + r"""]
        \centering
        """)

        if self.label:
            strings.append(r"\caption{" + self.caption + "}")
            strings.append(r"\label{tab:" + self.label + "}")

        if self.subtables > 1:
            subtables = self._subtables_split()
            width = "{:.3f}\linewidth".format(0.95 / self.subtables)

            for i, rows in enumerate(subtables):
                strings.append(r"\begin{{subtable}}[t]{{{0}}}%".format(width))
                strings.append(r"""
                \centering
                \vspace{0pt}
                """)
                self._latex_tabular(strings, rows)
                strings.append(r"\end{subtable}%")
                if i != len(subtables) - 1:
                    strings.append("\hfill%")

        else:
            rows = self._format_rows()
            self._latex_tabular(strings, rows)

        strings.append(r"""
        \end{table}
        """)
        return "\n".join(strings)

    def _latex_tabular(self, strings, rows):
        x = "|".join(["c"] * self.columns)
        strings.append(r"\begin{tabular}{|" + x + "|}")
        strings.append(r"\hline")

        if self.headings:
            latex = " & ".join(str(x) for x in self.headings)
            strings.append(latex + r" \\")
            strings.append(r"\hline")

        for row in rows:
            if row is self.VDOTS:
                row = [r"\vdots"] * self.columns
            latex = " & ".join(row)
            strings.append(latex + r" \\")

        strings.append(r"""
        \hline
        \end{tabular}%""")

    def _repr_html_(self):
        strings = []

        strings.append("""
        <style type="text/css">
        .util_Table td { text-align: center; }
        .util_Table tbody tr, .util_Table tbody td {
            border-bottom: 0;
            border-top: 0;
        }
        .util_Table_subtable {
            float: left;
        }
        </style>
        """)

        if self.label:
            c = self.caption
            l = "<code>[{}]</code>".format(self.label)

            strings.append("""
            <h3>{1} {2}</h3>
            """.format(self.columns, c, l))

        if self.subtables > 1:
            subtables = self._subtables_split()
            # width = 0.95 / self.subtables

            strings.append("<div class='clearfix'>")
            for rows in subtables:
                strings.append("<div class='util_Table_subtable'>")
                self._html_table(strings, rows)
                strings.append("</div>")
            strings.append("</div>")

        else:
            rows = self._format_rows()
            self._html_table(strings, rows)

        return "\n".join(strings)

    def _html_table(self, strings, rows):
        strings.append("<table class='util_Table'>")

        if self.headings:
            strings.append("<thead>")
            strings.append("<tr>")
            headings = map("<th>{0}</th>".format, self.headings)
            strings.append("\n".join(headings))
            strings.append("</tr>")
            strings.append("</thead>")

        strings.append("<tbody>")

        for row in rows:
            if row is self.VDOTS:
                row = ["\u22ee"] * self.columns

            strings.append("<tr>")
            row = map("<td>{0}</td>".format, row)
            strings.append("\n".join(row))
            strings.append("</tr>")

        strings.append("</tbody>")
        strings.append("</table>")

    def __repr__(self):
        if self.headings:
            widths = [len(x) for x in self.headings]
            data = [self.headings]
        else:
            widths = None
            data = []

        # don't forget - self._format_rows() is a generator that yields generators
        for row in self._format_rows():
            if row is self.VDOTS:
                continue

            r = list(row)
            w = [len(x) for x in r]

            if widths is None:
                widths = w
            else:
                widths = [max(a, b) for a, b in zip(widths, w)]

            data.append(list(r))

        strings = []
        if self.label:
            c = self.caption.replace("\n", " ")
            strings.append('Table: {0} ({1})'.format(self.label, c))

        for row in data:
            if row is self.VDOTS:
                strings.append('...')
            else:
                r = [x.ljust(b + 4) for x, b in zip(row, widths)]
                strings.append(''.join(r))

        return '\n'.join(strings)

    def __html__(self):
        return self._repr_html_()


class LatexFigure(object):

    extension = 'pdf'

    def __init__(self, label, caption, fig=None, position="", star=False,
                 options='width=\columnwidth', margin=False):
        """
        A LaTeX IPython DisplayObject Figure

        `label` is mandatory, since it also sets the filename. It will
        have ``fig:`` preprended to it.

        `fig` is optional - the current figure (via ``gcf``) will be used
        if it is not set.

        `position` is either the float placement specifier or the subfigure
        vertical position.

        If `subfigure` is set to true, a subfigure with width `width` will
        be created.

        The figure is saved (via ``savefig``) as a PDF file in the current
        directory.

        Displaying the object produces LaTeX (only) to embed the figure.
        A little hacky, but since this is meant for use in the notebook
        it is assumed that the figure is going to be displayed automatically
        in HTML independently.
        """
        if fig is None:
            from matplotlib.pyplot import gcf

            fig = gcf()

        self.label = label
        self.caption = caption
        self.fig = fig
        self.position = position
        self.options = options
        self.star = star
        self.margin = margin

        self.filename = "figure_{0:s}.{1:s}".format(label, self.__class__.extension)

        import pylab as plt

        try:
            plt.savefig(self.filename, bbox_inches='tight')
        except:
            plt.savefig(self.filename)

    def _repr_html_(self):
        # Bit crude. Hide ourselves to the notebook viewer, since we'll
        # have been shown already anyway.
        # Nicer solutions are afaict infeasible.
        return markdown2html('> **Figure (<a name="fig:{label:s}">{label:s}</a>)**: {caption:s}'.format(
            label=self.label, caption=self.caption))

    def _repr_latex_(self, subfigure=None):
        if subfigure:
            environment = "subfigure"
            args = "[{position}]{{{width}}}".format(**subfigure)
        else:
            environment = "figure"
            args = "[{0}]".format(self.position)
        args = args.replace('[]', '')

        if self.star:
            environment += '*'
        elif self.margin & (not subfigure):
            environment = "margin" + environment

        return r"""\begin{{{env:s}}}{args:s}
            \centering
            \includegraphics[{options:s}]{{{fname:s}}}
            \caption{{{caption:s}}}
            \label{{fig:{label:s}}}
            \end{{{env:s}}}
            """.format(env=environment, args=args, options=self.options,
                       fname=self.filename, caption=self.caption,
                       label=self.label)

    def __repr__(self):
        c = self.caption.replace("\n", " ")
        return "Figure: {0} ({1})".format(self.label, c)

    def __html__(self):
        return ""


class LatexSubfigures(object):
    def __init__(self, label, caption, figures, position='h',
                 subfigure_position='b', star=False):
        """
        Displays several :cls:`LatexFigures` as sub-figures, two per row.

        `figures` should be an array of :cls:`LatexFigure` objects, not
        :cls:`matplotlib.Figure` objects.
        """

        self.label = label
        self.caption = caption
        self.figures = figures
        self.position = position
        self.subfigure_position = subfigure_position
        self.star = star

    def _repr_html_(self):
        # Bit crude. Hide ourselves to the notebook viewer, since we'll
        # have been shown already anyway.
        # Nicer solutions are afaict infeasible.
        return markdown2html('> **Figure (<a name="fig:{label:s}">{label:s}</a>)**: {caption:s}'.format(
            label=self.label, caption=self.caption))

    def _repr_latex_(self):
        strings = []

        environment = "figure"
        if self.star:
            environment += '*'

        strings.append(r"""\begin{""" + environment + """}[""" + self.position + r"""]
            \centering
        """)

        #left = True
        #first = True
        opts = {"position": self.subfigure_position,
                "width": "{0:0.2f}\linewidth".format((1 - len(self.figures) * 0.01) / len(self.figures))}
        for f in self.figures:
            #if left and not first:
            #    strings.append(r"\vspace{1em}")

            # have to be quite careful about whitespace
            latex = f._repr_latex_(subfigure=opts).strip()

            #if left:
            #    latex += '%'
            #else:
            #    latex += r'\newline'
            #first = False
            #left = not left

            strings.append(latex)

        strings.append(r"""
            \caption{""" + self.caption + r"""}
            \label{fig:""" + self.label + r"""}
        \end{""" + environment + """}
        """)

        return "\n".join(strings)

    def __repr__(self):
        c = self.caption.replace("\n", " ")
        strings = ["Figure group: {0} ({1})".format(self.label, c)]
        strings += [repr(x) for x in self.figures]
        return "\n".join(strings)

    def __html__(self):
        return ""


class LatexNumberFormatter(object):
    """
    Format floats in exponent notation using latex markup for the exponent

    e.g., ``$-4.234 \\times 10^{-5}$``

    Usage:

    >>> fmtr = LatexNumberFormatter(sf=4)
    >>> fmtr(-4.234e-5)
    "$-4.234 \\\\times 10^{-5}$"
    """

    def __init__(self, sf=10):
        """Create a callable object that formats numbers"""
        self.sf = sf
        self.s_fmt = "{{:.{0}e}}".format(self.sf)

    def __call__(self, n):
        """Format `n`"""
        n = self.s_fmt.format(n)
        n, e, exp = n.partition("e")
        if e == "e":
            exp = int(exp)
            if not n.startswith("-"):
                n = r"\phantom{-}" + n
            return r"${} \times 10^{{{}}}$".format(n, exp)
        else:
            return "${}$".format(n)


"""
Simple progressbar
==================

This package implement a unique progress bar class that can be used to decorate
an iterator, a function or even standalone.

The format of the meter is flexible and can display along with the progress
meter, the running time, an eta, and the rate of the iterations.

An example is:
    description    [----------] k/n  10% [time: 00:00:00, eta: 00:00:00, 2.7 iters/sec]
"""


class NBPbar(object):
    """
    make a progress string  in a shape of:

    [----------] k/n  10% [time: 00:00:00, eta: 00:00:00, 2.7 iters/sec]

    Attributes
    ---------

    time: bool, optional (default: True)
        if set, add the runtime information

    eta: bool, optional (default: True)
        if set, add an estimated time to completion

    rate: bool, optional (default: True)
        if set, add the rate information

    length: int, optional (default: None)
        number of characters showing the progress meter itself
        if None, the meter will adapt to the buffer width

        TODO: make it variable with the buffer length

    keep: bool, optional (default: True)
        If not set, deletes its traces from screen after completion

    file: buffer
        the buffer to write into

    mininterval: float (default: 0.5)
        minimum time in seconds between two updates of the meter

    miniters: int, optional (default: 1)
        minimum iteration number between two updates of the meter

    units: str, optional (default: 'iters')
        unit of the iteration
    """
    def __init__(self, desc=None, maxval=None, time=True, eta=True, rate=True, length=None,
                 file=None, keep=True, mininterval=0.5, miniters=1, units='iters', **kwargs):
        self.time = time
        self.eta = eta
        self.rate = rate
        self.desc = desc or ''
        self.units = units
        self.file = file or sys.stdout
        self._last_print_len = 0
        self.keep = keep
        self.mininterval = mininterval
        self.miniters = miniters
        self._auto_width = True
        self.length = 10
        if length is not None:
            self.length = length
            self._auto_width = False
        # backward compatibility
        self._start_t = _time.time()
        self._maxval = maxval
        if 'txt' in kwargs:
            self.desc = kwargs['txt']
        self._F = None

    @staticmethod
    def format_interval(t):
        """ make a human readable time interval decomposed into days, hours,
        minutes and seconds

        Parameters
        ----------
        t: int
            interval in seconds

        Returns
        -------
        txt: str
            string representing the interval
            (format:  <days>d <hrs>:<min>:<sec>)
        """
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        d, h = divmod(h, 24)

        txt = '{m:02d}:{s:02d}'
        if h:
            txt = '{h:02d}:' + txt
        if d:
            txt = '{d:d}d ' + txt
        return txt.format(d=d, h=h, m=m, s=s)

    def build_str_meter(self, n, total, elapsed):
        """
        make a progress string  in a shape of:

            k/n  10% [time: 00:00:00, eta: 00:00:00, 2.7 iters/sec]

        Parameters
        ----------
        n: int
            number of finished iterations

        total: int
            total number of iterations, or None

        elapsed: int
            number of seconds passed since start

        Returns
        -------
        txt: str
            string representing the meter
        """
        if n > total:
            total = None

        vals = {'n': n}
        vals['elapsed'] = self.format_interval(elapsed)
        vals['rate'] = '{0:5.2f}'.format((n / elapsed)) if elapsed else '?'
        vals['units'] = self.units

        if not total:
            txt = '{desc:s} {n:d}'
        else:
            txt = '{desc:s} {n:d}/{total:d} {percent:s}'

        if self.time or self.eta or self.rate:
            txt += ' ['
            info = []
            if self.time:
                info.append('time: {elapsed:s}')
            if self.eta and total:
                info.append('eta: {left:s}')
            if self.rate:
                info.append('{rate:s} {units:s}/sec')
            txt += ', '.join(info) + ']'

        if not total:
            return txt.format(**vals)

        frac = float(n) / total
        vals['desc'] = self.desc
        vals['percent'] = '{0:3.0%}'.format(frac)
        vals['left'] = self.format_interval(elapsed / n * (total - n)) if n else '?'
        vals['total'] = total

        return txt.format(**vals)

    def print_status(self, n, total, elapsed):
        from IPython.html.widgets import FloatProgress
        desc = self.build_str_meter(n, total, elapsed)
        if self._F is None:
            self._F = FloatProgress(min=0, max=total, description=desc)
            display(self._F)

        self._F.value = n
        self._F.description = desc

    def iterover(self, iterable, total=None):
        """
        Get an iterable object, and return an iterator which acts exactly like the
        iterable, but prints a progress meter and updates it every time a value is
        requested.

        Parameters
        ----------
        iterable: generator or iterable object
            object to iter over.

        total: int, optional
            the number of iterations is assumed to be the length of the
            iterator.  But sometimes the iterable has no associated length or
            its length is not the actual number of future iterations. In this
            case, total can be set to define the number of iterations.

        Returns
        -------
        gen: generator
            pass the values from the initial iterator
        """
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = self._maxval

        self.print_status(0, total, 0)
        last_print_n = 0

        start_t = last_print_t = _time.time()

        for n, obj in enumerate(iterable):
            yield obj
            if n - last_print_n >= self.miniters:
                cur_t = _time.time()
                if cur_t - last_print_t >= self.mininterval:
                    self.print_status(n, total, cur_t - start_t)
                    last_print_n = n
                    last_print_t = cur_t

        if self.keep:
            if last_print_n < n:
                cur_t = _time.time()
                self.print_status(n, total, cur_t - start_t)
            self.file.write('\n')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    def update(self, n, desc=None, total=None):
        """ Kept for backward compatibility and the decorator feature """
        if total is None:
            total = self._maxval
        if desc is not None:
            self.desc = desc
        cur_t = _time.time()
        self.print_status(n, total, cur_t - self._start_t)
