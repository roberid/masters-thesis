%% This is western.cls, a class file in the making by Eric Tittley
%% revised by Marcel VanDalfsen, for use at McMaster University
\NeedsTeXFormat{LaTeX2e}[1996/06/01]
\ProvidesClass{macthesis}
	[2004/12/01 v1.1 McMaster University thesis document class]
%              [1998/11/01 v1.0 U of Western Ontario document class]

% -------USER REDEFINEABLE VARIABLES-----

% Font types and sizes for section titles
\newcommand{\partsize}{\Huge}
\newcommand{\partfont}{\bfseries}
\newcommand{\chaptersize}{\LARGE}
\newcommand{\chapterfont}{\bfseries}
\newcommand{\sectionsize}{\Large}
\newcommand{\sectionfont}{\bfseries}
\newcommand{\subsectionsize}{\normalsize}
\newcommand{\subsectionfont}{\bfseries}
\newcommand{\subsubsectionsize}{\normalsize}
\newcommand{\subsubsectionfont}{\it}
\newcommand{\paragraphsize}{\normalsize}
\newcommand{\paragraphfont}{\slseries}
\newcommand{\subparagraphsize}{\normalsize}
\newcommand{\subparagraphfont}{\rm}

% Lables for parts of the text
%  Pages
\newcommand\abstractname{Abstract}
\newcommand\acknowledgementname{Acknowledgements}
\newcommand\coauthorshipname{Co-Authorship}
\newcommand\appendixname{Appendix}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\listappendixname{List of Appendices}
\newcommand\contentsname{Table of Contents}
\newcommand\listequationname{List of Equations}
\newcommand\vitaname{Vita}
% Headings
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
%  Floats
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
% What to call a chapter in the TOC entry for a chapter.
\newcommand\ChapterTOCLabel{Chapter}
\newcommand\AppendixTOCLabel{Appendix}

% -----END USER DEFINEABLE VARIABLES-----

% ----OPTIONS----
% The font size
\DeclareOption{10pt}{\newcommand\@ptsize{0}}
\DeclareOption{11pt}{\newcommand\@ptsize{1}}
\DeclareOption{12pt}{\newcommand\@ptsize{2}}

% The line spacing
\DeclareOption{draft}{
 \setlength\overfullrule{5pt}
 \renewcommand{\baselinestretch}{1.35}
}
\DeclareOption{final}{
 \setlength\overfullrule{0pt}
 \renewcommand{\baselinestretch}{1.65}%was 1.20 in Marcel's version
}

% equation number format
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}

% two side or one side
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}

% Error when we get an option we don't recognize
\DeclareOption*{\typeout Unkown option passed to macthesis: \CurrentOption}

% DEFAULT BEHAVIOUR
\ExecuteOptions{12pt,final,oneside,fleqn}
\ProcessOptions

% ------END OPTIONS------

% -----PAPER DIMENSIONS------
\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
%\setlength\parskip{0\p@ \@plus \p@}
\setlength\parskip{6pt \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\clubpenalty10000
\@clubpenalty \clubpenalty
\widowpenalty10000
\displaywidowpenalty10000
\newcommand{\single}{\par\setlength{\baselineskip}{1\p@}}

\setlength\paperheight {11in}
\setlength\paperwidth  {8.5in}

% The top margin size (the addition of the next 3 lengths + 1 inch)
\setlength\voffset{0in} 
\setlength\topmargin{1.0in}
\setlength\headheight{12\p@}
\addtolength\topmargin{-\headheight}
\setlength\headsep{24\p@}
\addtolength\topmargin{-\headsep}
%\setlength\headsep{0pt}
%
% The bottom margin (the addition of the next length + 1 inch)
\setlength\footskip{0.4in}
% The left margin (the next 2 lengths + 1 inch)
\setlength\hoffset{0in}
\setlength\oddsidemargin{0.5in}
\setlength\evensidemargin{0.5in}% for double sided
%    %\setlength\hoffset{0.5in} \setlength\oddsidemargin{0pt}
% The Right side margin (addition of next 3)
\setlength\marginparsep{0pt} \setlength\marginparwidth{0pt}
\setlength\marginparpush{0pt}

% The dimensions of the text area (measured from the top right margin corner)
\setlength{\textheight}{8.2in}
\setlength{\textwidth}{5.5in}

% The page style.  We'll start with roman numerals (until the first chapter),
% a plain pagestyle, and and empty heading
\pagenumbering{roman}
\pagestyle{plain}
\markright{}
% -----End PAPER DIMENSIONS------

% -----FONTS--------
% Input the initialization file for the chosen font size and set
% some other parameters dependant on the font size
% 'Old' Fonts
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}

% -----END FONTS-----

% ----PAGE STYLES-----
% for \pagestyle{headings}
\def\ps@headings{%
    %\topmargin -45pt \headsep .5in \footskip .5in
    \let\@oddfoot\@empty
   \def\@oddhead{\hbox to \textwidth {\slshape\rightmark\hfil} \hbox to .5in {\thepage}}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ %
        \fi
        ##1}}}}

% for \pagestyle{myheadings}
\newfont{\ssdc}{cmssdc10}
\def\kmvrule{\leavevmode\cleaders \hb@xt@ .455em{--}\hfill\kern\z@}
\def\ps@myheadings{%
  %\topmargin -45pt \headsep .5in
  % what follows is original, mine is after that (KMV)
  %  \let\@oddfoot\@empty\let\@evenfoot\@empty
  %  \def\@evenhead{\thepage\hfil\slshape\leftmark}%
  %  \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
  %  \let\@mkboth\@gobbletwo
  %  \let\chaptermark\@gobble
  %  \let\sectionmark\@gobble
  % new header/footer from KMV (Marcel VanDalfsen)
  \def\@oddfoot{\hfil\thepage\hfil}%
  \def\@evenfoot{\hfil\thepage\hfil}%
  \def\@evenhead{\ssdc\shortdegree\ Thesis\ \kmvrule\ \firstname\ \middlename\ \lastname\ \kmvrule\ \theuniversity\ -\ \shortdepartment\ \kmvrule\ \submissionyear}%
  \let\@oddhead\@evenhead%
%  \def\@oddhead{Hey\hfil}%
  \let\@mkboth\@gobbletwo
  \let\chaptermark\@gobble
  \let\sectionmark\@gobble
}

%-----End PAGE STYLES-----

% -------SECTIONS-------

% Section initialization of counters and declaration of count formats
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
%\setcounter{tocdepth}{2}  % is set lower in document actually
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\newcommand\@chapapp{\chaptername}
\newcounter {appendix}
%\newcommand\@apdxapp{\appendixname}
\newcommand\part{%
  \clearpage
	\thispagestyle{myheadings}%
  \@tempswafalse
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \partsize\partfont \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \partsize\partfont #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \partsize\partfont #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
                \null
                \thispagestyle{empty}%
                \newpage
              \fi
             }
\newcommand\chapter{
	\clearpage
%	\thispagestyle{myheadings}% this is original
%	\pagestyle{headings}% this is fancy headings
	\global\@topnum\z@
	\@afterindentfalse
	\secdef\@chapter\@schapter
}
\def\@chapter[#1]#2{
 \ifnum \c@secnumdepth >\m@ne
  \refstepcounter{chapter}
  \addcontentsline{toc}{chapter}{\protect\numberline{\thechapter}#1}
 \else
  \addcontentsline{toc}{chapter}{#1}%
 \fi
 \chaptermark{#1}%
 \addtocontents{lof}{\protect\addvspace{10\p@}}%
 \addtocontents{lot}{\protect\addvspace{15\p@}}%
 \addtocontents{loe}{\protect\addvspace{10\p@}}%
 \@makechapterhead{#2}%
}

% This is the chapter command used by the actual chapters in the body
\def\@makechapterhead#1{%
  %\vspace*{-40\p@}%
	%comment out next line for no space between top of page and Chapter heading
	\vspace*{25\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \chaptersize\chapterfont \@chapapp\space \thechapter
        \par\nobreak
        %\vskip 0\p@
        \vskip 13\p@
    \fi
    \interlinepenalty\@M
    \chaptersize\chapterfont #1\par\nobreak
%    \vskip 10\p@
    \vskip 26\p@
  }}
\def\@schapter#1{
                   \@makeschapterhead{#1}%
                   \@afterheading
		%\addcontentsline{toc}{section}{\protect\bibname}
                }
% The chapter command used by things like the bibliography
\def\@makeschapterhead#1{%
  %\vspace*{50\p@}%
  \vspace*{-20\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \chaptersize\chapterfont  #1\par\nobreak
    %\vskip 0\p@
    \vskip 40\p@
  }}


%%--------
%% test for appendix section - Marcel
%\newcommand\appendix{\par
%  \setcounter{chapter}{0}
%  \setcounter{section}{0}%
%  \setcounter{subsection}{0}%
%  \gdef\ChapterTOCLabel{Appendix}%
%  \gdef\chaptername{Appendix}%
%  \gdef\@chapapp{\appendixname}%
%  \gdef\thechapter{\@Alph\c@chapter}
%%  \gdef\thesection{\@Alph\c@section}}
%}



%------------- below commented out by Marcel
%appendix 
\newcommand\appendix{
	\clearpage
	%\thispagestyle{myheadings}
	\renewcommand\@chapapp{\appendixname}%
	\renewcommand\theappendix{\@Alph\c@appendix}
	\global\@topnum\z@
	\@afterindentfalse
	\secdef\@appendix\@schapter
}

\def\@appendix[#1]#2{
	\refstepcounter{appendix}
	\addcontentsline{toc}{appendix}{\protect\numberline{\theappendix}#1}
	\chaptermark{#1}%
	\addtocontents{lof}{\protect\addvspace{10\p@}}%
	\addtocontents{lot}{\protect\addvspace{10\p@}}%
	%\addtocontents{loa}{\protect\addvspace{10\p@}}%
	\@makeappendixhead{#2}%
}
\def\@makeappendixhead#1{%
  \vspace*{-20\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \chaptersize\chapterfont \@chapapp\space \theappendix
        \par\nobreak
        \vskip 20\p@
    \fi
    \interlinepenalty\@M
    \chaptersize\chapterfont #1\par\nobreak
    \vskip 40\p@
	}
}
%------------------ above commented out

% appendix
\newcommand\appendices{\par
  	\setcounter{appendix}{0}%
	%\addcontentsline{toc}{preface}{APPENDICES}
}


% The next command tells how to put a appendix entry in the TOC
\newcommand*\l@appendix[2]{
 \pagebreak[3] 
 \vskip 1.0em plus 1pt \@tempdima 1.5em
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth 
  \parfillskip -\@pnumwidth 
%  \small\leavevmode{\bf #1}% KMV
  \small\leavevmode{\bf \AppendixTOCLabel~#1}% KMV
%  \small\leavevmode{\bf \ChapterTOCLabel\ - #1}  % GP
%  \nobreak\leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}
\hfill
\nobreak\hb@xt@\@pnumwidth
  {\hss\normalsize\bf #2}\par
 \endgroup
}



% The next command tells how to put a bibliography entry in the TOC
\newcommand*\l@bibliography[2]{
 \pagebreak[3] 
 \vskip 1.0em plus 1pt \@tempdima 1.5em
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth 
  \parfillskip -\@pnumwidth 
%  \small\leavevmode{\bf #1}% KMV
  \small\leavevmode{\bf #1}% KMV
%  \small\leavevmode{\bf \ChapterTOCLabel\ - #1}  % GP
%  \nobreak\leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}
\hfill
\nobreak\hb@xt@\@pnumwidth
  {\hss\normalsize\bf #2}\par
 \endgroup
}

%---------------------------------


% How sections ennumerations appear
\renewcommand\thepart{\@Roman\value{part}}
\renewcommand\thechapter{\@arabic\value{chapter}}
\renewcommand\thesection{\thechapter.\@arabic\value{section}}
\renewcommand\thesubsection{\thesection.\@arabic\value{subsection}}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\value{subsubsection}}
\renewcommand\theparagraph{\thesubsubsection.\@arabic\value{paragraph}}
\renewcommand\thesubparagraph{\theparagraph.\@arabic\value{subparagraph}}

% Definitions for the section declarations.  Increment the counter for the section,
% add a line to the TOC, and print the formatted line in the text (section #, Label)
\newcommand{\newsection}[2]{
\vskip 30\p@
\stepcounter{#1}
%\vskip -20\p@
%{\noindent\normalfont\csname #1size\endcsname\csname #1font\endcsname\csname the#1\endcsname\space #2\\*}
{\noindent\normalfont\csname #1size\endcsname\csname #1font\endcsname\csname the#1\endcsname\space #2}
\def\@currentlabel{\csname the#1\endcsname}
\nopagebreak
\addcontentsline{toc}{#1}{\protect\numberline{\csname the#1\endcsname}#2}
}
\newcommand{\section}[1]{\newsection{section}{#1}}
\newcommand{\subsection}[1]{\newsection{subsection}{#1}}
\newcommand{\subsubsection}[1]{\newsection{subsubsection}{#1}}
\newcommand{\paragraph}[1]{\newsection{paragraph}{#1}}
\newcommand{\subparagraph}[1]{\newsection{subparagraph}{#1}}

%I don't know what these do
\setlength\leftmargini  {2.5em}
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\setlength\leftmarginv  {1em}
\setlength\leftmarginvi {1em}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@highpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}

% -----END SECTION DEFINITIONS------

% -----Environments----------------

% Description
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}

% Verse
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}

% Quotation
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}


% Quote
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}


% titlepage, used by \thesistitle
\newenvironment{titlepage}
    {
      \newpage
      \thispagestyle{empty}
      \setcounter{page}\@ne
    }
    {
     \newpage
     \setcounter{page}\@ne
    }

% appendix
%\newcommand\appendix{\par
%  \setcounter{chapter}{0}%
%  \setcounter{section}{0}%
%  \renewcommand\@chapapp{\appendixname}%
%  \renewcommand\thechapter{\@Alph\c@chapter}
%}

% ----The following stuff is used to reset the baseline stretch within floats.
%     It was stolen from doublespace.sty (Stephen Page et al, 1986)
%     I don't understand any of it, but it works.  Any side effects?
%     Perhaps I should just use doublespace as a global environment.
%     Anyway, I stripped out lots of stuff which didn't seem to be
%     relevant.  This may have been a mistake if that stuff was there for
%     a reason in the first place.
%     --- Eric

%---Reset baselinestretch within footnotes and floats. Originally stolen
%   from Stanford thesis style.
\def\@newbaseline{
    \ifx\undefined\setnew@baselineskip
      \baselineskip\f@baselineskip\relax
    \else
      \setnew@baselineskip
    \fi
    \baselineskip\baselinestretch\baselineskip
    \setbox\strutbox\hbox{\vrule\@height.7\baselineskip
    \@depth.3\baselineskip \@width\z@}
    \normalbaselineskip\baselineskip
 }

\def\@xfloat#1[#2]{
 \ifhmode \@bsphack\@floatpenalty-\@Mii
 \else \@floatpenalty-\@Miii
 \fi
 \def\@captype{#1}
 \ifinner \@parmoderr\@floatpenalty\z@
 \else
  \@next\@currbox\@freelist{
   \@tempcnta\csname ftype@#1\endcsname
   \multiply\@tempcnta\@xxxii\advance\@tempcnta\sixt@@n
   \@tfor \@tempa :=#2\do{
    \if\@tempa h\advance\@tempcnta \@ne\fi
    \if\@tempa t\advance\@tempcnta \tw@\fi
    \if\@tempa b\advance\@tempcnta 4\relax\fi
    \if\@tempa p\advance\@tempcnta 8\relax\fi
   }
   \global\count\@currbox\@tempcnta
  }
  \@fltovf
 \fi
 \global\setbox\@currbox\vbox\bgroup 
 \def\baselinestretch{1}
 \@newbaseline %include new baselinestretch
 \boxmaxdepth\z@
 \hsize\columnwidth \@parboxrestore
}
%  End of Robbery from doublespace.
%  ---Eric

% Stuff to set up limits on the number of floats per page
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}

% More stuff needed by floats?
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand{\thefigure}{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}{\par\@float{figure}}{\end@float}
\newenvironment{figure*}{\par\@dblfloat{figure}}{\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}{\par\@float{table}}{\end@float}
\newenvironment{table*}{\par\@dblfloat{table}}{\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
 \vskip\abovecaptionskip
 \sbox\@tempboxa{#1: #2}%
 \ifdim \wd\@tempboxa >\hsize
   #1: #2\par
 \else
   \global \@minipagefalse
   \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
 \fi
 \vskip\belowcaptionskip
}

% The following stuff is for the theBibliography (Do we need this?)
%\newdimen\bibindent
%\setlength\bibindent{1.5em}
%\newenvironment{thebibliography}[1]
%     {\chapter*{\bibname
%        \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}}%
%      \list{\@biblabel{\@arabic\c@enumiv}}%
%           {\settowidth\labelwidth{\@biblabel{#1}}%
%            \leftmargin\labelwidth
%            \advance\leftmargin\labelsep
%            \@openbib@code
%            \usecounter{enumiv}%
%            \let\p@enumiv\@empty
%            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
%      \sloppy
%      \clubpenalty4000
%      \@clubpenalty \clubpenalty
%      \widowpenalty4000%
%      \sfcode`\.\@m}
%     {\def\@noitemerr
%       {\@latex@warning{Empty `thebibliography' environment}}%
%      \endlist}

%\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
%\let\@openbib@code\@empty

% The index (Do we need this?)
%\newenvironment{theindex}
%                {
%                \columnseprule \z@
%                \columnsep 35\p@
%                \twocolumn[\@makeschapterhead{\indexname}]%
%                \@mkboth{\MakeUppercase\indexname}%
%                        {\MakeUppercase\indexname}%
%                \thispagestyle{myheadings}\parindent\z@
%                \parskip\z@ \@plus .3\p@\relax
%                \let\item\@idxitem}
%               {\clearpage}

% acknowledgement environment by E.Tittley (981006)
\newenvironment{acknowledgement}{
 \addcontentsline{toc}{preface}{\protect\acknowledgementname}
 \newpage
 \thispagestyle{plain}
 \small
 \begin{center}
  {\chaptersize\chapterfont \acknowledgementname}
 \end{center}
}
{\clearpage}
%end of acknowledgement environment 

% coauthorship environment by E.Tittley (990531)
\newenvironment{coauthorship}{
 \addcontentsline{toc}{preface}{\protect\coauthorshipname}
 \newpage
 \thispagestyle{plain}
 \small
 \begin{center}
  {\chaptersize\chapterfont \coauthorshipname}
 \end{center}
}
{\clearpage}
%end of coauthorship environment 


% vita environment
% The vita environment sets up the page
% The vitaentry environment allows for a list of Vita information
% The vitapubs environment allows for a list of Vita publication information
%\newcommand{\vitaentrylabel}[1]{\parbox[t][\height][t]{1.5in}{#1}\hfil}
\newcommand{\vitaentrylabel}[1]
{\raisebox{0pt}[0ex][0pt]{\makebox[\labelwidth][l]
 {\parbox[t]{\labelwidth}{\setlength{\baselineskip}{5pt}\hspace{0pt}#1}}}}
\newenvironment{vitaentry}
 {
  \begin{list}{}
  {\renewcommand{\makelabel}{\vitaentrylabel}
   \setlength{\itemindent}{0in}
   \setlength{\labelwidth}{2.5in}
   \setlength{\labelsep}{0.2in}
   \setlength{\leftmargin}{3in}
   \setlength{\parsep}{0.7cm}
  }
 }
 {\end{list}}

%Vitapub environment
\newcommand{\heading}[1]{{\noindent\bf #1}}
\newenvironment{vitapub}
 {\renewcommand{\labelenumi}{(\theenumi)}
  \begin{enumerate}
 }
{\end{enumerate}}

\newenvironment{vita}{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\vitaname}
 \single
 \begin{center}
  {\chaptersize\chapterfont \vitaname}
 \end{center}
 \vspace{20pt}
}
{\clearpage}


%end of vita environment

% work environment, modified by E.Tittley (981006) (What is this for?)
%\newenvironment{work}{
%\begingroup
%  \halign\bgroup\parbox[t]{2in}{\obeylines{##}\vspace{0.1in}}&
%          \parbox[t]{4in}{##}\vspace{0.1in}\cr
%}
%{\egroup\endgroup}
%end of work environment

% abstract environment
\newenvironment{abstract}{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\abstractname}
 \begin{center}
  {\chaptersize\chapterfont \abstractname}
 \end{center}
}
{\clearpage}
%end of abstract environment

%----END ENVIRONMENTS----

%----COMMANDS------------

% \tableofcontents generates a TOC page
% The next few sections are for the table of contents
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
% increase the next to creat a table with more depths.  That is,
% a depth of 1 will only put sections in the TOC while a depth of
% 5 will put everything down to subparagraphs
\setcounter{tocdepth}{2}


\newcommand\tableofcontents{%
	\clearpage
%    \chapter*{\contentsname
%        \@mkboth{%
%           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
%    \addcontentsline{toc}{preface}{\protect\contentsname}%
%	\chapter*{\contentsname}
	\begin{center}{\bf\LARGE\contentsname}\end{center}

	\vspace{0.7em}
    \@starttoc{toc}%
    }



%\newcommand\tableofcontents{
% \clearpage
% \addcontentsline{toc}{preface}{\protect\contentsname}
% \begin{center}
%  {\chaptersize\chapterfont \contentsname}
% \end{center}
% {\raggedleft\normalsize Page \par}
% \@starttoc{toc}
% \clearpage
%}

% The next command tells how to put a "part" entry in the TOC
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \setlength\@tempdima{3em}%
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}

% The next command tells how to put a preface entry in the TOC
\newcommand*\l@preface[2]{
 \pagebreak[3] 
% \vskip 1.0em plus 1pt \@tempdima 1.5em
 \begingroup
%  \noindent\small\leavevmode{\sc \MakeUppercase#1}
  \noindent\small\leavevmode{\bf #1}     % GP
%  \nobreak\leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}
\hfill
\nobreak\hb@xt@\@pnumwidth
 {\hss\normalsize #2}\par
\endgroup
}

% The next command tells how to put a chapter entry in the TOC
\newcommand*\l@chapter[2]{
 \pagebreak[3] 
 \vskip 1.0em plus 1pt \@tempdima 1.5em
 \begingroup
  \parindent \z@ \rightskip \@pnumwidth 
  \parfillskip -\@pnumwidth 
%  \small\leavevmode{\bf #1}% KMV
  \small\leavevmode{\bf \ChapterTOCLabel~#1}% KMV
%  \small\leavevmode{\bf \ChapterTOCLabel\ - #1}  % GP
%  \small\leavevmode{\sc \ChapterTOCLabel\ - #1}  % GP
%  \nobreak\leaders\hbox{$\m@th \mkern \@dotsep mu\hbox{.}\mkern \@dotsep mu$}
\hfill
\nobreak\hb@xt@\@pnumwidth
  {\hss\normalsize\bf #2}\par
 \endgroup
}

% The next commands tell how to put the respective entries in the TOC
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}

% \listoffigures generates the "List of Figures" page
\newcommand\listoffigures{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\listfigurename}
 \begin{center}
  {\chaptersize\chapterfont \listfigurename}
 \end{center}
% {\normalsize Figure \hfill Description \hfill Page \par}
 \@starttoc{lof}
 \clearpage
}
% The next command tells how each entry should look in the List of Figures
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}

% \listoftables generates the "List of Tables" page
\newcommand\listoftables{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\listtablename}
 \begin{center}
  {\chaptersize\chapterfont \listtablename}
 \end{center}
% {\normalsize Table \hfill Description \hfill Page \par}
 \@starttoc{lot}
 \label{frontmatterpages}
 \cleardoublepage
}

% The next alias says to make the table entries in the List of Tables look
% the same as the figure entries in the List of Figures
\let\l@table\l@figure

% experiment - make a "list of equations" page
\newcommand\listofequations{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\listequationname}
 \begin{center}
  {\chaptersize\chapterfont \listequationname}
 \end{center}
 \@starttoc{loe}
 \clearpage
}
\let\l@equation\l@figure

\newcommand{\equationcaption}[1]{
%\def\@currentlabel{\csname the#1\endcsname}
%\nopagebreak
\addcontentsline{loe}{equation}{\protect\numberline{\csname theequation\endcsname}#1}
}


% \listofappendices generates the "List of Appendices" page
% Todd Fuller 990618
\newcommand\listofappendices{
 \clearpage
 \addcontentsline{toc}{preface}{\protect\listappendixname}
 \begin{center}
  {\chaptersize\chapterfont \listappendixname}
 \end{center}
 {\normalsize Appendix \hfill Description \hfill Page \par}
 \@starttoc{loa}
 \clearpage
}

% I don't know what the next few lines are for.  Index?
%\newcommand\@idxitem{\par\hangindent 40\p@}
%\newcommand\subitem{\@idxitem \hspace*{20\p@}}
%\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
%\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}

\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\today{}
\edef\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}

% A generic testing condition used below
\def\ifundefined#1{\expandafter\ifx\csname#1\endcsname\relax} % TeX book
                                                              % excercise 7.7

%Generate the half title page   - Glen Petitpas
%  this page is also sometimes called the 'bastard title page'
%  in professional bookbinding
\newcommand{\halftitlepage}{%
~
%\thispagestyle{empty}
\pagestyle{empty}
 \vspace{5cm}
 \begin{center}
  {\sc\halftitle}
 \end{center}
\cleardoublepage
}

%Generate the title page, Modified to McMaster Standards by Glen Petitpas
% need : \firstname, \lastname, \thetitle, \thedegree, \currentmonth
%        \currentyear, \department
%
% Note; 'fulfillment' is American 'fulfilment' is British/Canadian
%
% \thesistitle should be all caps
\newcommand{\thesistitle}{%
\setcounter{page}{1}
\begin{titlepage}
~
 \vspace{1cm}
 \begin{center}
                 {\sc\large\thetitle}\\
  \vspace{2.5cm} By\\
                 {\sc \firstname~\middlename~\lastname,~\degrees}\\
  \vspace{4.0cm} A Thesis\\[0.25ex]
                 Submitted to the School of Graduate Studies\\[0.25ex]
                 in Partial Fulfilment of the Requirements\\[0.25ex]
                 for the Degree\\[0.25ex]
                 \thedegree\\
  \vfill
  \theuniversity\\[0.25ex]
  \ifundefined{bindingyear}\def\bindingyear{\submissionyear}\fi
  \copyright Copyright by~\firstname~\lastname,~\submissionmonth~\bindingyear
 \end{center}
\end{titlepage}
\clearpage
}

%Generate a dedication
\newcommand{\dedication}[1]{
\clearpage
\thispagestyle{plain}\vspace*{1.5in}\begin{center}\it
%\thispagestyle{empty}\vspace*{1.5in}\begin{center}\it
\shortstack[r]{
#1
}
\end{center}
\clearpage}

% descriptive notes
\newcommand{\descriptivenotes}{%
%\pagestyle{myheadings}% for dorky university header/footer
\pagestyle{plain}% for 'normal' pages (with pg# on bottom)
\setcounter{page}{2}
\addcontentsline{toc}{preface}{\protect Descriptive Notes}
~
\vspace{1cm}
\\
{\sc \thedegree} (\submissionyear) \hfill \theuniversity\\[0.25ex]
(\department) \hfill  \thelocation \\
\\
TITLE: ~~~\thetitle \\[0.25ex]
AUTHOR:  ~~~\firstname~\lastname, \degrees ~(\degreefrom)\\[0.25ex]
SUPERVISORS: ~~~\supervisor \\[0.25ex]
NUMBER OF PAGES: ~~~\firstpages, \lastpages

\clearpage
}

%Generate a certificate
% This requires one argument which is either: \projectcert,\certone,\certtwo,\phdcert
% There is slightly different output, primarily in the sigitures areal,
% dependant on this option's value.
% The following is the "enumeration" of the arguments.  I could have 1,2,3,4 or
%  w,d,t,v.  So long as they are unique and one character long.
\newcommand{\projectcert}{A}
\newcommand{\certone}{B}
\newcommand{\certtwo}{C}
\newcommand{\phdcert}{D}
\newcommand{\certificate}[1]{
\thispagestyle{plain}
\addcontentsline{toc}{preface}{\protect Certificate of Examination}
\begin{center}
  McMASTER UNIVERSITY\\
  SCHOOL OF GRADUATE STUDIES\\
  \vspace{20pt}
  CERTIFICATE OF EXAMINATION
\end{center}
\newcommand{\MDD}{\hspace{0.5in}}
\newcommand{\LDD}{\makebox[2.75in]}
\newcommand{\RDD}{\makebox[0.5in]{}\makebox[2.75in]}
\newlength{\sigheight}\setlength{\sigheight}{0.7cm}
\vspace*{\sigheight}
\if#1\projectcert
 \noindent\LDD{Chief Advisor\hfill}\RDD{Examining Board\hfill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vfill
\fi
\if#1\certone
 \noindent\LDD{Chief Advisor\hfill}\RDD{Examining Board\hfill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hfill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hfill}\RDD{\hrulefill}\par
 \vspace{\sigheight}
\fi
\if #1\certtwo
 \noindent\LDD{Advisors\hfill}\RDD{Examining Board\hfill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hfill}\RDD{\hrulefill}\par
 \vspace{\sigheight}
\fi
\if#1\phdcert
 \noindent\LDD{Chief Advisor\hfill}\RDD{Examining Board\hfill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{Advisory Committee\hfill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
 \noindent\LDD{\hrulefill}\RDD{\hrulefill}\par
 \vspace*{\sigheight}
\fi
\begin{center}
\if#1\projectcert
  The project by\\
\else  
  The thesis by\\
\fi
  \firstname~\lastname\\
  \vspace*{0.1in}
  entitled\\
  {\sc\thetitle}\\
  \vspace*{15pt}
  is accepted in partial fulfilment of the\\
  requirements for the degree of\\
  \thedegree
\end{center}
\vfill
\noindent\LDD[l]{Date \hrulefill}\RDD{\hrulefill}\par
\noindent\LDD{}\RDD{\hfill Chair of Examining Board\hfill}\par
\clearpage
}
%end of \certificate command

% -----MISC----
% a protected reference command
\newcommand{\refer}{\protect\ref}
%These do what?
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}

% ---END MISC----


% MIKE REID ADDED THIS TO PREVENT THE "COMMAND NOT DEFINED" ERROR
% WHEN LATEX PARSES NATBIB.STY, WHICH INCLUDES THE DIRECTIVE
% "RENEWENVIRONMENT{THEBIBLIOGRAPHY}".  THIS SEEMS TO HAVE TO BE
% PRECEEDED BY A \DEF\THEBIBLIOGRAPHY, SO I SPLICED THIS ONE IN.
% I AM NOT INTENDING TO USE THIS CODE FOR ANY OTHER REASON.
%    *             BIBLIOGRAPHY             *
%    ****************************************
%
% The thebibliography environment executes the following commands:
%
%  \def\newblock{\hskip .11em plus .33em minus .07em} --
%      Defines the `closed' format, where the blocks (major units of
%      information) of an entry run together.
%
%  \sloppy  -- Used because it's rather hard to do line breaks in
%      bibliographies,
%
%  \sfcode`\.=1000\relax --
%      Causes a `.' (period) not toproduce an end-of-sentence space.

\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}

%% RmS 91/10/27 [ .. ] replaced by \@biblabel{ .. }
%% RmS 91/11/13: Changed counter enumi to enumiv,
%%               as it says in the comment in latex.tex
%% RmS 92/01/14: Set \p@enumiv to {} and \theenumiv to \arabic{enumiv}
%%               to get correct references
\def\thebibliography#1{\chapter*{\bibname\@mkboth
  {\uppercase{\bibname}}{\uppercase{\bibname}}}
  \addcontentsline{toc}{chapter}{\bibname}
  \list{\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \def\newblock{\hskip .11em plus.33em minus.07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=\@m}


%% 91/08/26 FMI & RmS: introduced warning instead of error

\def\endthebibliography{%
  \def\@noitemerr{\@warning{Empty `thebibliography' environment}}%
  \endlist}


% \def\@biblabel#1{[#1]\hfill}  % Produces the label for a \bibitem[...]
                                % command.
% \def\@cite#1{[#1]}            % Produces the output of the \cite
                                % command.


% Fix up indentation of footnotes

\makeatletter
\newlength{\myFootnoteWidth}
\newlength{\myFootnoteLabel}
\setlength{\myFootnoteLabel}{1.2em}%  <-- can be changed to any valid value
\renewcommand{\@makefntext}[1]{%
  \setlength{\myFootnoteWidth}{\columnwidth}%
  \addtolength{\myFootnoteWidth}{-\myFootnoteLabel}%
  \noindent\makebox[\myFootnoteLabel][r]{\@makefnmark\ }%
  \parbox[t]{\myFootnoteWidth}{#1}%
}
\makeatother

\endinput
%%
%% End of file `macthesis.cls'.
