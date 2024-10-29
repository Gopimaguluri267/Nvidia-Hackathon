# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.modify import Modify

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class SectionNumberFormatter(DocumentModifier):
    def modify_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats section numbers in legal text.
        """
        if isinstance(document, str):
            text = document
        elif isinstance(document, dict) and "question" in document:
            text = document["question"]
        else:
            return document

        # Format section numbers
        text = re.sub(r'§(\S)', r'§ \1', text)
        text = re.sub(r'§§(\S)', r'§§ \1', text)
        text = re.sub(r'(\d+[a-z]?)-(\d+)', r'\1–\2', text)
        text = re.sub(r'(\d+)([a-z]{1,3})\b', r'\1\2', text)
        text = re.sub(r'(\d+[a-z]{2})-(\d+)', r'\1–\2', text)
        text = re.sub(r'(§§? \d+)-(\d+)', r'\1–\2', text)
        text = re.sub(r'([a-z]{2})-(\d+)', r'\1–\2', text)
        text = re.sub(r'\bUSC\b', 'U.S.C.', text)
        text = re.sub(r'§(\d)', r'§ \1', text)
        text = re.sub(r'§§(\d)', r'§§ \1', text)
        text = re.sub(r'(§§? \d+)-(\d+)', r'\1–\2', text)
        text = re.sub(r'\(([a-z])\)', r'(\1)', text)
        text = re.sub(r'\(([a-z])\)-\(([a-z])\)', r'(\1)–(\2)', text)
        text = re.sub(r'(\d+)\s+U\.S\.C\.', r'\1 U.S.C.', text)
        text = re.sub(r'(\d+)\s+U\.S\.C\.\s+§\s+(\d+)(\([a-z]\)(?:-\([a-z]\))?)?', 
                      lambda m: f"{m.group(1)} U.S.C. § {m.group(2)}{m.group(3) if m.group(3) else ''}", 
                      text)

        if isinstance(document, str):
            return text
        document["question"] = text
        return document


def redact_pii(dataset : DocumentDataset, text_field):
    pii_redactor = Modify(
        PiiModifier(
            supported_entities=[
                "ADDRESS",
                "EMAIL_ADDRESS",
                "LOCATION",
                "PERSON",
                "URL",
                "PHONE_NUMBER",
            ],
            anonymize_action="replace",
            device="cpu",
        ),
        text_field = text_field,
    )
    return pii_redactor(dataset)
