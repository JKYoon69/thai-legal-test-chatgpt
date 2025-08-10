# -*- coding: utf-8 -*-
from .parser import detect_doc_type, parse_document
from .postprocess import validate_tree, make_chunks, guess_law_name
from .schema import ParseResult, Node, Chunk
