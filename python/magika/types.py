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

"""Type definitions for the Magika file type detection library."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class MagikaStatus(str, Enum):
    """Status codes for Magika detection results."""

    OK = "ok"
    EMPTY_FILE = "empty"
    TOO_SHORT = "too_short"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ContentTypeInfo:
    """Metadata about a detected content type."""

    label: str
    """Short label for the content type (e.g., 'python', 'pdf')."""

    mime_type: str
    """MIME type string (e.g., 'text/x-python', 'application/pdf')."""

    group: str
    """High-level group the content type belongs to (e.g., 'code', 'document')."""

    description: str
    """Human-readable description of the content type."""

    extensions: list[str] = field(default_factory=list)
    """Common file extensions associated with this content type."""

    is_text: bool = False
    """Whether this content type is considered a text format."""

    def __str__(self) -> str:
        return f"{self.label} ({self.mime_type})"


@dataclass
class MagikaResult:
    """Result of a Magika file type detection operation."""

    path: Optional[Path]
    """Path to the file that was analyzed, or None if bytes were provided directly."""

    dl: ContentTypeInfo
    """Content type as determined by the deep-learning model."""

    output: ContentTypeInfo
    """Final output content type, which may incorporate heuristics or overrides."""

    score: float
    """Confidence score for the detection result, in the range [0.0, 1.0]."""

    status: MagikaStatus = MagikaStatus.OK
    """Status code indicating success or the reason for a non-standard result."""

    @property
    def ok(self) -> bool:
        """Return True if the detection completed without errors."""
        return self.status == MagikaStatus.OK

    def __str__(self) -> str:
        path_str = str(self.path) if self.path else "<bytes>"
        return (
            f"MagikaResult(path={path_str!r}, label={self.output.label!r}, "
            f"mime_type={self.output.mime_type!r}, score={self.score:.4f}, "
            f"status={self.status.value!r})"
        )


@dataclass
class ModelFeatures:
    """Input features extracted from file content for the ML model."""

    beg: list[int]
    """Byte values from the beginning of the file."""

    mid: list[int]
    """Byte values from the middle of the file."""

    end: list[int]
    """Byte values from the end of the file."""
