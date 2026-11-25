#!/usr/bin/env python3
"""
Gutenberg Top Authors Downloader
Downloads all ENGLISH books from Project Gutenberg's top N authors (last 30 days) for style transfer research.

Usage:
    python gutenberg_downloader.py --authors 100 --output ./gutenberg_corpus
    
"""

import os
import json
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from time import sleep
from tqdm import tqdm
from datetime import datetime
import logging
import re

class GutenbergAuthorDownloader:
    def __init__(self, output_dir="gutenberg_authors_corpus", log_level=logging.INFO):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.authors_metadata = {}
        self.base_url = "https://www.gutenberg.org"
        
        # Setup logging
        self.setup_logging(log_level)
        
        # Progress tracking
        self.progress_file = self.output_dir / 'progress.json'
        self.completed_authors = self.load_progress()
        
        # Non-English author names/keywords to filter out
        self.non_english_patterns = [
            'garnett', 'dostoyevsky', 'dumas', 'maquet', 'nietzsche',
            'verne', 'tolstoy', 'balzac', 'doré', 'homer', 'hugo',
            'burton', 'rizal', 'dante', 'goethe', 'ibsen', 'chekhov',
            'cervantes', 'maupassant', 'morley', 'levy', 'voltaire',
            'wagner',  'zola', 'frazer', 'grimm', 'kafka',
            'honig', 'yonge', 'smollett'
            # Plato  'meredith',
        ]
        
    def setup_logging(self, log_level):
        """Setup logging configuration"""
        log_file = self.output_dir / 'download.log'
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_progress(self):
        """Load progress from previous runs"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                return set(progress.get('completed_authors', []))
            except:
                return set()
        return set()
    
    def save_progress(self, author_id):
        """Save progress after completing each author"""
        self.completed_authors.add(author_id)
        with open(self.progress_file, 'w') as f:
            json.dump({'completed_authors': list(self.completed_authors)}, f)
    
    def is_likely_english_author(self, author_name):
        """Check if author name suggests they write in English"""
        name_lower = author_name.lower()
        
        # Check against non-English patterns
        for pattern in self.non_english_patterns:
            if pattern in name_lower:
                return False
        
        # Additional checks for obvious non-English
        non_english_indicators = [
            'graf', 'von', 'de', 'der', 'la', 'le', 'ibn', 'al-'
        ]
        
        for indicator in non_english_indicators:
            if f' {indicator} ' in f' {name_lower} ' or name_lower.startswith(indicator + ' '):
                return False
        
        return True
    
    def scrape_top_authors_from_gutenberg(self, n=100):
        """Scrape actual top authors from Project Gutenberg's top authors page"""
        url = f"{self.base_url}/browse/scores/top"
        self.logger.info(f"Scraping top {n} authors from: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            authors = []
            
            # Find the "Top 100 Authors last 30 days" section
            # Look for the heading
            authors_section = None
            for heading in soup.find_all(['h2', 'h3']):
                if 'authors' in heading.text.lower() and 'last' in heading.text.lower():
                    authors_section = heading
                    break
            
            if not authors_section:
                self.logger.warning("Could not find authors section, trying alternative parsing")
                return self.parse_authors_alternative(soup, n)
            
            # Find the ordered list after the heading
            ol = authors_section.find_next('ol')
            
            if ol:
                for li in ol.find_all('li', limit=n * 2):  # Get extra to filter
                    # Find author link
                    author_link = li.find('a', href=re.compile(r'/ebooks/author/\d+'))
                    
                    if author_link:
                        href = author_link['href']
                        # Extract author ID
                        match = re.search(r'/ebooks/author/(\d+)', href)
                        
                        if match:
                            author_id = match.group(1)
                            author_name = author_link.text.strip()
                            
                            # Extract download count from parentheses
                            downloads = 0
                            download_match = re.search(r'\((\d+)\)', li.text)
                            if download_match:
                                downloads = int(download_match.group(1))
                            
                            # Filter for likely English authors
                            if self.is_likely_english_author(author_name):
                                authors.append({
                                    'id': author_id,
                                    'name': author_name,
                                    'downloads': downloads,
                                    'url': f"{self.base_url}/ebooks/author/{author_id}"
                                })
                                
                                if len(authors) >= n:
                                    break
            
            self.logger.info(f"Scraped {len(authors)} English-speaking authors")
            return authors
            
        except Exception as e:
            self.logger.error(f"Failed to scrape authors: {e}")
            return []
    
    def parse_authors_alternative(self, soup, n):
        """Alternative parsing method if main method fails"""
        self.logger.info("Using alternative parsing method")
        
        authors = []
        
        # Find all author links
        for link in soup.find_all('a', href=re.compile(r'/ebooks/author/\d+')):
            match = re.search(r'/ebooks/author/(\d+)', link['href'])
            
            if match:
                author_id = match.group(1)
                author_name = link.text.strip()
                
                if author_name and len(author_name) > 2:
                    # Filter for English authors
                    if self.is_likely_english_author(author_name):
                        if author_id not in [a['id'] for a in authors]:
                            authors.append({
                                'id': author_id,
                                'name': author_name,
                                'downloads': 0,
                                'url': f"{self.base_url}/ebooks/author/{author_id}"
                            })
                            
                            if len(authors) >= n:
                                break
        
        return authors
    
    def get_top_authors(self, n=100):
        """Get top N English-speaking authors from Project Gutenberg"""
        self.logger.info(f"Fetching top {n} English-speaking authors")
        
        # Scrape from actual website
        authors = self.scrape_top_authors_from_gutenberg(n)
        
        if not authors:
            self.logger.warning("Scraping failed, falling back to curated list")
            # Fallback to curated list
            authors = self.get_fallback_authors()[:n]
            for author in authors:
                author['url'] = f"{self.base_url}/ebooks/author/{author['id']}"
        
        self.logger.info(f"Prepared {len(authors)} English-speaking authors for download")
        
        # Log top 10
        if authors:
            self.logger.info("\nTop 10 authors to download:")
            for i, author in enumerate(authors[:10], 1):
                downloads = f" ({author['downloads']} downloads)" if author.get('downloads') else ""
                self.logger.info(f"  {i:2d}. {author['name']}{downloads}")
        
        return authors
    
    def get_fallback_authors(self):
        """Fallback list of known English authors if scraping fails"""
        return [
            {'id': '65', 'name': 'Shakespeare, William'},
            {'id': '37', 'name': 'Dickens, Charles'},
            {'id': '53', 'name': 'Twain, Mark'},
            {'id': '61', 'name': 'Shelley, Mary Wollstonecraft'},
            {'id': '68', 'name': 'Austen, Jane'},
            {'id': '69', 'name': 'Doyle, Arthur Conan'},
            {'id': '35', 'name': 'Stevenson, Robert Louis'},
            {'id': '1961', 'name': 'Widger, David'},
            {'id': '9', 'name': 'Melville, Herman'},
            {'id': '36', 'name': 'Montgomery, L. M.'},
            {'id': '30', 'name': 'Wells, H. G.'},
            {'id': '102', 'name': 'Alcott, Louisa May'},
            {'id': '7', 'name': 'Carroll, Lewis'},
            {'id': '481', 'name': 'Poe, Edgar Allan'},
            {'id': '80', 'name': 'Chesterton, G. K.'},
            {'id': '42', 'name': 'Baum, L. Frank'},
            {'id': '90', 'name': 'Eliot, George'},
            {'id': '59', 'name': 'Scott, Walter'},
            {'id': '326', 'name': 'Swift, Jonathan'},
            {'id': '975', 'name': 'Forster, E. M.'},
            {'id': '132', 'name': 'Kipling, Rudyard'},
            {'id': '113', 'name': 'James, Henry'},
            {'id': '120', 'name': 'London, Jack'},
            {'id': '451', 'name': 'Christie, Agatha'},
            {'id': '125', 'name': 'Conrad, Joseph'},
            {'id': '408', 'name': 'Brontë, Charlotte'},
            {'id': '420', 'name': 'Fitzgerald, F. Scott'},
            {'id': '220', 'name': 'Gaskell, Elizabeth Cleghorn'},
            {'id': '23', 'name': 'Hardy, Thomas'},
            {'id': '405', 'name': 'Brontë, Emily'},
            {'id': '54', 'name': 'Thoreau, Henry David'},
            {'id': '34', 'name': 'Irving, Washington'},
            {'id': '1039', 'name': 'Joyce, James'},
            {'id': '190', 'name': 'Stoker, Bram'},
            {'id': '324', 'name': 'Trollope, Anthony'},
            {'id': '783', 'name': 'Wodehouse, P. G.'},
            {'id': '104', 'name': 'Wharton, Edith'},
            {'id': '38191', 'name': 'Chambers, Robert W.'},
            {'id': '1032', 'name': 'Henty, G. A.'},
            {'id': '945', 'name': 'Yonge, Charlotte M.'},
            {'id': '205', 'name': 'Wheatley, Henry B.'},
            {'id': '520', 'name': 'Meredith, George'},
            {'id': '359', 'name': 'Ruskin, John'},
            # Jowett, Benjamin, 1817-1893 94
            {'id': '27', 'name': 'Gilman, Charlotte Perkins'},
            {'id': '783', 'name': 'Wodehouse, P. G.'},
            {'id': '2683', 'name': 'Lee, Vernon'},
            {'id': '1959', 'name': 'Richardson, Samuel'},
            {'id': '89', 'name': 'Virginia Woolf'},
            {'id': '111', 'name': 'Wilde, Oscar'},
            {'id': '28', 'name': 'Hawthorne, Nathaniel'},
            # George Orwell 
]
    
    def detect_text_language(self, text):
        """Simple heuristic to detect if text is English"""
        english_indicators = [
            'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 
            'this', 'but', 'from', 'they', 'which', 'their', 'would'
        ]
        
        # Take a sample from the middle of the text
        sample_size = min(5000, len(text))
        start = max(0, (len(text) - sample_size) // 2)
        sample = text[start:start + sample_size].lower()
        
        # Count English indicator words
        count = sum(sample.count(f' {word} ') for word in english_indicators)
        
        return count > 20
    
    def get_author_books(self, author_url, author_name):
        """Get all book IDs for a specific author"""
        self.logger.info(f"Fetching books for: {author_name}")
        
        try:
            response = requests.get(author_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            books = []
            seen_ids = set()
            
            # Find all book links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/ebooks/' in href and '/ebooks/author/' not in href:
                    match = re.search(r'/ebooks/(\d+)', href)
                    if match:
                        book_id = int(match.group(1))
                        
                        if book_id not in seen_ids:
                            seen_ids.add(book_id)
                            
                            title = link.text.strip()
                            if not title or len(title) < 2:
                                parent = link.parent
                                if parent:
                                    title = parent.get_text().strip()
                            
                            if not title:
                                title = f"Book {book_id}"
                            
                            books.append({
                                'id': book_id,
                                'title': title
                            })
            
            self.logger.info(f"Found {len(books)} books for {author_name}")
            return books
            
        except Exception as e:
            self.logger.error(f"Error fetching books for {author_name}: {e}")
            return []
    
    def download_book(self, book_id):
        """Download a single book by ID"""
        urls = [
            f"{self.base_url}/files/{book_id}/{book_id}-0.txt",
            f"{self.base_url}/files/{book_id}/{book_id}.txt",
            f"{self.base_url}/cache/epub/{book_id}/pg{book_id}.txt",
            f"{self.base_url}/ebooks/{book_id}.txt.utf-8",
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    try:
                        text = response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            text = response.content.decode('latin-1', errors='ignore')
                        except:
                            text = response.content.decode('utf-8', errors='ignore')
                    
                    if self.detect_text_language(text):
                        return text
                    else:
                        self.logger.debug(f"Book {book_id} does not appear to be English")
                        return None
                        
            except Exception as e:
                continue
        
        return None
    
    def clean_filename(self, name):
        """Clean author name for use as directory name"""
        cleaned = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.', ','))
        cleaned = cleaned.strip().replace(' ', '_')
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        return cleaned[:100]
    
    def strip_gutenberg_header_footer(self, text):
        """Remove Project Gutenberg header and footer from text"""
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
            "***START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT",
        ]
        
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of the Project Gutenberg",
            "End of Project Gutenberg",
            "***END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT",
        ]
        
        start_pos = 0
        for marker in start_markers:
            pos = text.upper().find(marker.upper())
            if pos != -1:
                start_pos = text.find('\n', pos) + 1
                break
        
        end_pos = len(text)
        for marker in end_markers:
            pos = text.upper().find(marker.upper())
            if pos != -1:
                end_pos = pos
                break
        
        cleaned_text = text[start_pos:end_pos].strip()
        
        if len(cleaned_text) < 1000 and len(text) > 1000:
            return text
        
        return cleaned_text
    
    def download_author_corpus(self, author_info):
        """Download all ENGLISH books for a single author"""
        author_name = author_info['name']
        author_id = author_info['id']
        author_url = author_info['url']
        downloads = author_info.get('downloads', 0)
        
        if author_id in self.completed_authors:
            self.logger.info(f"Skipping {author_name} (already completed)")
            return
        
        author_dir = self.output_dir / self.clean_filename(author_name)
        author_dir.mkdir(exist_ok=True)
        
        books = self.get_author_books(author_url, author_name)
        
        if not books:
            self.logger.warning(f"No books found for {author_name}")
            self.save_progress(author_id)
            return
        
        downloaded_books = []
        failed_books = []
        non_english_books = []
        
        for book in tqdm(books, desc=f"  {author_name[:30]}", leave=False):
            book_id = book['id']
            title = book['title']
            
            filepath = author_dir / f"book_{book_id}.txt"
            if filepath.exists():
                file_size = filepath.stat().st_size
                downloaded_books.append({
                    'id': book_id,
                    'title': title,
                    'filepath': str(filepath.relative_to(self.output_dir)),
                    'length': file_size
                })
                continue
            
            text = self.download_book(book_id)
            if text:
                text = self.strip_gutenberg_header_footer(text)
                
                if not self.detect_text_language(text):
                    self.logger.debug(f"Skipping non-English book: {title}")
                    non_english_books.append({'id': book_id, 'title': title})
                    continue
                
                if len(text) > 1000:
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        downloaded_books.append({
                            'id': book_id,
                            'title': title,
                            'filepath': str(filepath.relative_to(self.output_dir)),
                            'length': len(text)
                        })
                    except Exception as e:
                        self.logger.error(f"Error saving {book_id}: {e}")
                        failed_books.append({'id': book_id, 'title': title})
                else:
                    self.logger.warning(f"Book {book_id} too short after cleaning")
                    failed_books.append({'id': book_id, 'title': title})
            else:
                failed_books.append({'id': book_id, 'title': title})
            
            sleep(1.5)
        
        self.authors_metadata[author_id] = {
            'name': author_name,
            'id': author_id,
            'downloads': downloads,
            'url': author_url,
            'directory': str(author_dir.relative_to(self.output_dir)),
            'books': downloaded_books,
            'failed_books': failed_books,
            'non_english_books': non_english_books,
            'total_books': len(downloaded_books),
            'total_failed': len(failed_books),
            'total_non_english': len(non_english_books),
            'download_date': datetime.now().isoformat()
        }
        
        with open(author_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.authors_metadata[author_id], f, indent=2, ensure_ascii=False)
        
        self.save_progress(author_id)
        
        self.logger.info(
            f"Completed {author_name}: {len(downloaded_books)} English books, "
            f"{len(non_english_books)} non-English, {len(failed_books)} failed"
        )
    
    def download_all(self, n_authors=100):
        """Download English books for top N English-speaking authors"""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting download of top {n_authors} English-speaking authors")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info("Filtering: ENGLISH BOOKS ONLY")
        self.logger.info("=" * 80)
        
        authors = self.get_top_authors(n_authors)
        
        if not authors:
            self.logger.error("No authors found! Exiting.")
            return
        
        remaining_authors = [a for a in authors if a['id'] not in self.completed_authors]
        
        self.logger.info(f"Total authors: {len(authors)}")
        self.logger.info(f"Already completed: {len(self.completed_authors)}")
        self.logger.info(f"Remaining: {len(remaining_authors)}")
        self.logger.info("=" * 80)
        
        for i, author in enumerate(tqdm(remaining_authors, desc="Authors"), 1):
            try:
                self.download_author_corpus(author)
                sleep(2)
                
            except KeyboardInterrupt:
                self.logger.warning("\n\nInterrupted by user! Progress saved.")
                self.logger.info("Run the script again to resume from where you left off.")
                break
                
            except Exception as e:
                self.logger.error(f"Error processing {author['name']}: {e}")
                continue
        
        self.save_final_metadata()
    
    def save_final_metadata(self):
        """Save complete metadata and summary"""
        metadata_file = self.output_dir / 'all_authors_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.authors_metadata, f, indent=2, ensure_ascii=False)
        
        total_books = sum(info['total_books'] for info in self.authors_metadata.values())
        total_failed = sum(info['total_failed'] for info in self.authors_metadata.values())
        total_non_english = sum(info.get('total_non_english', 0) for info in self.authors_metadata.values())
        
        summary = {
            'download_date': datetime.now().isoformat(),
            'language_filter': 'English only',
            'source': 'Project Gutenberg Top 100 Authors (last 30 days)',
            'total_authors': len(self.authors_metadata),
            'total_books_downloaded': total_books,
            'total_books_failed': total_failed,
            'total_non_english_filtered': total_non_english,
            'authors': [
                {
                    'name': info['name'],
                    'downloads': info.get('downloads', 0),
                    'books_count': info['total_books'],
                    'failed_count': info['total_failed'],
                    'non_english_count': info.get('total_non_english', 0)
                }
                for info in sorted(
                    self.authors_metadata.values(),
                    key=lambda x: x['total_books'],
                    reverse=True
                )
            ]
        }
        
        with open(self.output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DOWNLOAD COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"Authors processed: {len(self.authors_metadata)}")
        self.logger.info(f"Total ENGLISH books downloaded: {total_books}")
        self.logger.info(f"Total non-English books filtered: {total_non_english}")
        self.logger.info(f"Total books failed: {total_failed}")
        self.logger.info(f"Metadata saved to: {metadata_file}")
        self.logger.info(f"Summary saved to: {self.output_dir / 'summary.json'}")
        self.logger.info("=" * 80)
        
        if summary['authors']:
            self.logger.info("\nTop 10 Authors by English Book Count:")
            for i, author in enumerate(summary['authors'][:10], 1):
                self.logger.info(
                    f"  {i:2d}. {author['name']}: {author['books_count']} books"
                )


def main():
    parser = argparse.ArgumentParser(
        description='Download ENGLISH books from Project Gutenberg top English-speaking authors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download top 100 English authors
  python gutenberg_downloader.py --authors 100 --output ./corpus
  
  # Download top 50 English authors with debug logging
  python gutenberg_downloader.py --authors 50 --output ./corpus --verbose
        """
    )
    
    parser.add_argument(
        '--authors', '-n',
        type=int,
        default=100,
        help='Number of top English-speaking authors to download (default: 100)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./gutenberg_authors_corpus',
        help='Output directory path (default: ./gutenberg_authors_corpus)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    downloader = GutenbergAuthorDownloader(
        output_dir=args.output,
        log_level=log_level
    )
    
    downloader.download_all(n_authors=args.authors)


if __name__ == "__main__":
    main()
