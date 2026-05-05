#!/usr/bin/env python3
"""
Corpus Curation Script for GraphRAG Lab
Fetches Wikipedia articles about tech companies and prepares them for entity extraction.
"""

import wikipedia
import time
import re
from pathlib import Path
from typing import List, Dict
import json

class CorpusCurator:
    def __init__(self, output_dir: str = "corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Metadata tracking
        self.metadata = []

    def fetch_articles(self, titles: List[str]) -> List[Dict]:
        """Fetch Wikipedia articles and return metadata."""
        articles = []

        for title in titles:
            try:
                print(f"Fetching: {title}")
                page = wikipedia.page(title, auto_suggest=False)

                article = {
                    "title": title,
                    "url": page.url,
                    "content": page.content,
                    "categories": page.categories[:5],  # First 5 categories
                    "length": len(page.content)
                }
                articles.append(article)

                # Save raw
                safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename = self.output_dir / f"{safe_title}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# {title}\n")
                    f.write(f"# Source: {page.url}\n\n")
                    f.write(page.content)

                time.sleep(1)  # Be polite to Wikipedia

            except wikipedia.exceptions.DisambiguationError as e:
                print(f"  ✗ Disambiguation error for {title}: trying alternative...")
                try:
                    # Try to get the first disambiguation option
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    article = {
                        "title": e.options[0],
                        "url": page.url,
                        "content": page.content,
                        "categories": page.categories[:5],
                        "length": len(page.content)
                    }
                    articles.append(article)
                    safe_title = e.options[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = self.output_dir / f"{safe_title}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# {e.options[0]}\n")
                        f.write(f"# Source: {page.url}\n\n")
                        f.write(page.content)
                    print(f"  ✓ Saved as: {e.options[0]}")
                    time.sleep(1)
                except Exception as e2:
                    print(f"  ✗ Failed completely: {e2}")
            except wikipedia.exceptions.PageError:
                print(f"  ✗ Page not found: {title}")
            except Exception as e:
                print(f"  ✗ Error fetching {title}: {e}")

        return articles

    def clean_text(self, text: str) -> str:
        """Remove Wikipedia-specific noise and format cleanly."""
        # Remove section headers (== Section ==)
        text = re.sub(r'=+\s*([^=]+)\s*=+', '', text)

        # Remove citation brackets [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[\w+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)

        # Remove entire sections
        sections_to_remove = [
            r'See also.*?(?=\n\n|\Z)',
            r'References.*?(?=\n\n|\Z)',
            r'External links.*?(?=\n\n|\Z)',
            r'Further reading.*?(?=\n\n|\Z)',
            r'Notes.*?(?=\n\n|\Z)',
            r'Bibliography.*?(?=\n\n|\Z)',
            r'Footnotes.*?(?=\n\n|\Z)',
            r'Portal.*?(?=\n\n|\Z)',
        ]
        for pattern in sections_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def process_corpus(self, articles: List[Dict]) -> List[Dict]:
        """Clean all articles and track metadata."""
        processed = []

        for article in articles:
            cleaned = self.clean_text(article["content"])

            processed_article = {
                "title": article["title"],
                "url": article["url"],
                "content": cleaned,
                "original_length": article["length"],
                "cleaned_length": len(cleaned),
                "valid": len(cleaned) > 200  # Minimum length check
            }
            processed.append(processed_article)

            # Save cleaned version
            safe_title = article["title"].replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = self.output_dir / f"cleaned_{safe_title}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {article['title']}\n")
                f.write(f"# Source: {article['url']}\n\n")
                f.write(cleaned)

        return processed

    def generate_manifest(self, articles: List[Dict], filename: str = "manifest.json"):
        """Create a manifest file with corpus metadata."""
        manifest = {
            "corpus_name": "Tech Company Corpus",
            "description": "Wikipedia articles about AI and tech companies for GraphRAG lab",
            "total_documents": len(articles),
            "documents": articles,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        manifest_path = self.output_dir / filename
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Manifest saved to: {manifest_path}")
        print(f"✓ Total documents: {len(articles)}")
        print(f"✓ Valid documents: {sum(a['valid'] for a in articles)}")

        return manifest


def main():
    # Define target entities - AI/tech companies and related topics
    TITLES = [
        # AI Companies (Primary)
        "OpenAI", "Anthropic", "DeepMind", "Google DeepMind",
        "Hugging Face", "Stability AI", "Cohere", "Inflection AI",
        "Mistral AI", "Character.AI", "Runway ML", "Replicate",
        "Together AI", "Hugging Face", "Adept", "AI21 Labs",

        # Big Tech AI Divisions
        "Google", "Microsoft", "Meta", "Tesla", "Apple", "Amazon",
        "NVIDIA", "Intel", "AMD", "Qualcomm", "IBM", "Oracle",
        "Salesforce", "Adobe",

        # Cloud/Infrastructure
        "Amazon Web Services", "Microsoft Azure", "Google Cloud Platform",
        "Databricks", "Snowflake", "MongoDB", "Confluent", "Elastic",
        "HashiCorp", "GitHub", "GitLab",

        # Related Tech
        "Artificial intelligence", "Machine learning", "Deep learning",
        "Large language model", "Transformer (machine learning model)",
        "Computer vision", "Natural language processing", "Robotics",

        # People mentioned in company articles (for entity richness)
        "Sam Altman", "Elon Musk", "Satya Nadella", "Sundar Pichai",
        "Tim Cook", "Jensen Huang", "Demis Hassabis", "Dario Amodei",
        "Mustafa Suleyman", "Ilya Sutskever", "Andrew Ng", "Yann LeCun",
    ]

    # Run curation
    curator = CorpusCurator(output_dir="corpus")

    print("=" * 60)
    print("CORPUS CURATION FOR GRAPHRAG LAB")
    print("=" * 60)

    print("\n1. Fetching Wikipedia articles...")
    raw_articles = curator.fetch_articles(TITLES)

    if len(raw_articles) == 0:
        print("\n⚠ No articles fetched! Trying with more flexible search...")
        # Fallback: search for topics
        search_terms = [
            "artificial intelligence companies", "AI startups",
            "machine learning companies", "tech companies"
        ]
        for term in search_terms:
            try:
                results = wikipedia.search(term, results=5)
                print(f"\nSearch '{term}' found: {results}")
                additional = curator.fetch_articles(results[:3])
                raw_articles.extend(additional)
            except Exception as e:
                print(f"Search failed for '{term}': {e}")

    print(f"\nFetched {len(raw_articles)} articles")

    print("\n2. Cleaning and preprocessing...")
    processed_articles = curator.process_corpus(raw_articles)

    print("\n3. Generating manifest...")
    manifest = curator.generate_manifest(processed_articles)

    print("\n" + "=" * 60)
    print("CORPUS CURATION COMPLETE")
    print("=" * 60)

    # Print summary statistics
    total_words = sum(a["cleaned_length"] for a in processed_articles)
    avg_words = total_words // len(processed_articles) if processed_articles else 0

    print(f"\nSummary:")
    print(f"  Total documents: {len(processed_articles)}")
    print(f"  Total words: {total_words:,}")
    print(f"  Average words per doc: {avg_words:,}")
    print(f"  Corpus directory: {curator.output_dir.absolute()}")


if __name__ == "__main__":
    main()
