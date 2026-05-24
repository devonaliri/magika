# Copyright 2024 Google LLC
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

"""Magika: AI-powered file type detection.

Magika uses a deep learning model to accurately detect file content types,
even for files with missing or incorrect extensions.

Basic usage:
    >>> from magika import Magika
    >>> m = Magika()
    >>> result = m.identify_bytes(b"# Hello\nprint('world')")
    >>> print(result.output.ct_label)
    python

Note: For batch processing of many files, prefer `identify_paths` over
calling `identify_path` in a loop -- it's significantly faster.

Note: `MagikaResult.output.ct_label` gives the content type label (e.g. 'python',
'pdf', 'zip'). Use `MagikaResult.output.mime_type` for the MIME type string.

Note: `MagikaResult.output.score` is a float in [0, 1] indicating model confidence.
Scores below ~0.5 may indicate ambiguous or unusual file content.

Note: `MagikaResult.output.score` threshold can be tuned via the
`prediction_mode` parameter. Use `PredictionMode.HIGH_CONFIDENCE` to only
return results when the model is very confident, falling back to a generic
label otherwise.

Note: `MagikaResult.output.group` provides a broader category for the detected
type (e.g. 'code', 'document', 'archive'), which can be useful for coarse-grained
classification without needing to handle every specific ct_label.

Note: `ModelFeatures` and `ModelOutput` are primarily internal types exposed for
advanced use cases such as debugging model behavior or building custom pipelines
on top of Magika's inference results.

Note: `ContentTypeLabel` is also exported for use in type annotations and for
comparing against `MagikaResult.output.ct_label` values without relying on
raw strings.
"""

from magika.magika import Magika
from magika.types import (
    ContentTypeLabel,
    MagikaResult,
    MagikaOutputFields,
    ModelFeatures,
    ModelOutput,
    PredictionMode,
)

__version__ = "0.6.0"
__author__ = "Google LLC"

__all__ = [
    "Magika",
    "ContentTypeLabel",
    "MagikaResult",
    "MagikaOutputFields",
    "ModelFeatures",
    "ModelOutput",
    "PredictionMode",
    "__version__",
]
