from bs4 import BeautifulSoup
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class WebExtractor:
    def __init__(self):
        self.important_tags = [
            'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'article', 'section', 'main', 'div', 'li',
            'td', 'th', 'dt', 'dd', 'figcaption'
        ]
        self.metadata_tags = {
            'title': 'Title',
            'meta': {
                'name': ['description', 'keywords', 'author'],
                'property': ['og:title', 'og:description']
            }
        }

    def _extract_metadata(self, soup: BeautifulSoup) -> List[str]:
        """Extract metadata from HTML"""
        metadata = []

        # Extract title
        if soup.title:
            metadata.append(f"Title: {soup.title.string.strip()}")

        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            property = meta.get('property', '')
            content = meta.get('content', '')

            if name in self.metadata_tags['meta']['name']:
                metadata.append(f"{name.title()}: {content}")
            elif property in self.metadata_tags['meta']['property']:
                metadata.append(f"{property.split(':')[-1].title()}: {content}")

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace and normalize line endings
        text = ' '.join(text.split())
        return text.strip()

    def _extract_structured_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract content with structure preservation"""
        content = []

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Process important elements in order of appearance
        for tag in soup.find_all(self.important_tags):
            text = self._clean_text(tag.get_text())
            if text:
                if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content.append(f"\n{tag.name.upper()}: {text}")
                elif tag.name == 'li':
                    content.append(f"• {text}")
                elif tag.name in ['td', 'th']:
                    content.append(f"[{text}]")
                else:
                    content.append(text)

        return content

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content with improved structure preservation"""
        try:
            # Parse HTML with lxml parser for better performance
            soup = BeautifulSoup(html_content, 'lxml')

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Extract main content
            content = self._extract_structured_content(soup)

            # Combine metadata and content
            result = '\n'.join(metadata)
            result += '\n\nContent:\n'
            result += '\n'.join(content)

            return result

        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            raise Exception(f"Error extracting text from HTML: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Error extracting text from HTML: {str(e)}")

    def extract_text_from_url(self, url):
        """Extract text from a web page URL"""
        try:
            # Fetch web page
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract text from HTML
            return self.extract_text_from_html(response.text)
            
        except Exception as e:
            raise Exception(f"Error extracting text from URL: {str(e)}")
