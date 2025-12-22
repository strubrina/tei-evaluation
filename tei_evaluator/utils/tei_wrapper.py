import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any

class TEIWrapper:
    """
    Simple utility class to wrap XML content with minimal TEI structure
    """

    def wrap_if_needed(self, xml_content: str,
                       title: str = "Temporary TEI Wrapping",
                       author: Optional[str] = None) -> str:
        """
        Test if XML root is 'text' or 'body' and wrap accordingly

        Args:
            xml_content: XML content to potentially wrap
            title: Document title for the header
            author: Optional author name

        Returns:
            str: Either the original content (if already TEI) or wrapped content
        """
        # Check if already TEI
        if self._is_tei_xml(xml_content):
            return xml_content

        # Check if root element is 'body' - needs double wrapping
        if self._has_body_root(xml_content):
            # Wrap body in text element first
            xml_content = f"""<text>
    {xml_content}
</text>"""

        # Check if root element is 'text' (or now wrapped body)
        if not self._has_text_root(xml_content):
            raise ValueError("XML content must have 'text' or 'body' as root element to be wrapped")

        # Create minimal teiHeader
        author_elem = f"<author>{author}</author>" if author else ""

        # Create wrapper with minimal teiHeader for TEI validation
        wrapped_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title>{title}</title>
                {author_elem}
            </titleStmt>
            <publicationStmt>
                <p>Unpublished</p>
            </publicationStmt>
            <sourceDesc>
                <p>No source description</p>
            </sourceDesc>
        </fileDesc>
    </teiHeader>
    {xml_content}
</TEI>"""

        return wrapped_content

    def _is_tei_xml(self, content: str) -> bool:
        """
        Check if content is already TEI XML by looking for TEI root element
        """
        try:
            root = ET.fromstring(content)
            local_name = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            return local_name.lower() == 'tei'
        except ET.ParseError:
            return False

    def _has_text_root(self, content: str) -> bool:
        """
        Check if XML content has 'text' as root element
        """
        try:
            root = ET.fromstring(content)
            local_name = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            return local_name.lower() == 'text'
        except ET.ParseError:
            return False

    def _has_body_root(self, content: str) -> bool:
        """
        Check if XML content has 'body' as root element
        """
        try:
            root = ET.fromstring(content)
            local_name = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            return local_name.lower() == 'body'
        except ET.ParseError:
            return False

    def process_file(self,
                    input_file: str,
                    output_file: Optional[str] = None,
                    title: Optional[str] = None,
                    author: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a file: wrap with TEI if needed

        Args:
            input_file: Path to input XML file
            output_file: Path to output file (if None, will add _tei suffix)
            title: Document title (if None, uses filename)
            author: Document author
            metadata: Dictionary with metadata (title, author, etc.) - takes precedence over individual params

        Returns:
            str: Path to the output file
        """
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Set up output file path
        if output_file is None:
            output_path = input_path.parent / f"{input_path.stem}_tei.xml"
        else:
            output_path = Path(output_file)

        # Read file content
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            with open(input_path, 'r', encoding='latin-1') as f:
                content = f.read().strip()

        # Handle metadata parameter (takes precedence over individual params)
        if metadata:
            title = metadata.get('title', title)
            author = metadata.get('author', author)

        # Use filename as title if not provided
        if title is None:
            title = input_path.stem.replace('_', ' ').title()

        # Wrap if needed
        try:
            wrapped_content = self.wrap_if_needed(content, title=title, author=author)

            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(wrapped_content)

            if wrapped_content == content:
                print(f"File {input_file} was already TEI XML, copied to {output_path}")
            else:
                print(f"Wrapped XML file: {output_path}")

            return str(output_path)

        except ValueError as e:
            print(f"Error processing {input_file}: {e}")
            raise

    def batch_process(self,
                     input_dir: str,
                     output_dir: Optional[str] = None,
                     pattern: str = "*.xml",
                     default_author: Optional[str] = None) -> list:
        """
        Process multiple files in a directory

        Args:
            input_dir: Input directory path
            output_dir: Output directory path (if None, uses input_dir)
            pattern: File pattern to match (default: *.xml)
            default_author: Default author to apply to all files

        Returns:
            list: List of output file paths
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path

        # Find matching files
        input_files = list(input_path.glob(pattern))

        if not input_files:
            print(f"No files found matching pattern '{pattern}' in {input_dir}")
            return []

        output_files = []

        print(f"Processing {len(input_files)} files...")

        for input_file in input_files:
            try:
                # Create output file path
                output_file = output_path / f"{input_file.stem}_tei.xml"

                # Process file
                result = self.process_file(
                    str(input_file),
                    str(output_file),
                    author=default_author
                )
                output_files.append(result)

            except (IOError, OSError, ValueError, RuntimeError) as e:
                print(f"Error processing {input_file}: {e}")

        print(f"Successfully processed {len(output_files)} files")
        return output_files


# Example usage
def example_usage():
    """Example of how to use the simple TEI wrapper"""

    # Create wrapper instance
    wrapper = TEIWrapper()

    # Example 1: Simple text element
    xml_content = """<text>
    <body>
        <div type="letter">
            <p>Dear John,</p>
            <p>I hope this letter finds you well.</p>
            <p>Best regards,<lb/>Jane</p>
        </div>
    </body>
</text>"""

    # Wrap if needed
    result = wrapper.wrap_if_needed(
        xml_content,
        title="Letter to John",
        author="Jane Smith"
    )

    print("Wrapped XML:")
    print(result)

if __name__ == "__main__":
    example_usage()