# Corpus Curation for GraphRAG Lab

## Overview

This guide covers how to curate the **Tech Company Corpus** for your GraphRAG implementation. The corpus should contain documents rich in entities (companies, people, technologies) and relationships (founded, worked_at, developed, invested, etc.).

---

## Target Corpus Specifications

**Based on lab materials**: 100 Wikipedia articles about AI companies

**Ideal Document Characteristics**:
- ✅ Rich in named entities (people, companies, products, technologies)
- ✅ Contains multiple relationship types
- ✅ Spans multiple documents (enables cross-document relationships)
- ✅ Text quality: well-structured, minimal noise
- ✅ Size: ~500-2000 words per document

---

## Recommended Data Sources

### Option 1: Wikipedia API (Recommended)

**Pros**: Free, high-quality, structured, easy to scrape
**Cons**: Rate limits, requires cleanup

#### Target Article Categories:
- AI companies (OpenAI, Anthropic, DeepMind, Google AI, etc.)
- Tech companies (Microsoft, Meta, Tesla, etc.)
- Founders/executives (Sam Altman, Elon Musk, Satya Nadella, etc.)
- Technology domains (Transformer, LLM, Computer Vision, etc.)

#### Python Script to Fetch Articles:

```python
import wikipedia
import time
from typing import List

# List of target entities
TECH_COMPANIES = [
    "OpenAI", "Anthropic", "DeepMind", "Google", "Microsoft",
    "Meta", "Tesla", "NVIDIA", "Apple", "Amazon",
    "Neuralink", "SpaceX", "Stripe", "GitHub", "Red Hat",
    "Cisco", "Intel", "AMD", "Qualcomm", "Broadcom",
    "Databricks", "Snowflake", "Palantir", "MongoDB", "Elastic",
    "Hugging Face", "GitHub", "Replicate", "Modal", "Runway"
]

FOUNDERS_PEOPLE = [
    "Sam Altman", "Elon Musk", "Satya Nadella", "Sundar Pichai",
    "Tim Cook", "Jensen Huang", "Marc Benioff", "Dustin Moskovitz",
    "Ciriath", "Dario Amodei", "Demis Hassabis", "Mustafa Suleyman"
]

def fetch_wikipedia_articles(titles: List[str], save_dir: str = "corpus"):
    """Fetch Wikipedia articles and save as text files."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            content = page.content

            # Save to file
            filename = f"{save_dir}/{title.replace(' ', '_').replace('/', '_')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(content)

            print(f"✓ Saved: {title}")
            time.sleep(1)  # Rate limiting

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"✗ Disambiguation error for {title}: {e.options[:3]}")
        except wikipedia.exceptions.PageError:
            print(f"✗ Page not found: {title}")
        except Exception as e:
            print(f"✗ Error fetching {title}: {e}")

if __name__ == "__main__":
    all_titles = TECH_COMPANIES + FOUNDERS_PEOPLE
    fetch_wikipedia_articles(all_titles)
```

---

### Option 2: HuggingFace Datasets

**Pros**: Pre-cleaned, structured format
**Cons**: May need domain-specific augmentation

**Potential datasets**:
- `wikipedia` (the full Wikipedia corpus)
- Custom company-specific datasets

```python
from datasets import load_dataset

# Load Wikipedia
ds = load_dataset("wikipedia", "20220301.en", split="train")

# Filter for tech/AI companies
# Requires a list of target company names
```

---

### Option 3: Manual Curation

If you have access to specific company documents, press releases, or internal knowledge bases, you can manually curate:

1. Copy-paste relevant articles
2. Use web scrapers (BeautifulSoup, Scrapy)
3. Export from Confluence/Notion/etc.

---

## Corpus Preprocessing Pipeline

After collecting raw text, apply these preprocessing steps:

### Step 1: Clean the Text

```python
import re

def clean_wikipedia_text(text: str) -> str:
    """Remove Wikipedia-specific noise."""
    # Remove section headers like "== References =="
    text = re.sub(r'==+.*?==+', '', text)

    # Remove citation brackets [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Remove "See also" sections
    text = re.sub(r'See also.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)

    # Remove external links
    text = re.sub(r'External links.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)

    # Remove "References" sections
    text = re.sub(r'References.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text
```

### Step 2: Document Chunking (Optional)

For large documents, split into manageable chunks (~500-1000 tokens):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(cleaned_text)
```

**Note**: For GraphRAG, you may want to keep documents whole or use larger chunks since entity relationships may span paragraphs.

### Step 3: Quality Filtering

```python
def validate_document(text: str, min_length: int = 200) -> bool:
    """Check if document meets quality criteria."""
    if len(text) < min_length:
        return False

    # Check for minimum entity density (estimate)
    # Count potential proper nouns (capitalized words)
    proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    # Should have at least 5 named entities for meaningful graph
    if proper_nouns < 5:
        return False

    return True
```

---

## Complete Corpus Setup Script

Here's a complete script that fetches, cleans, and prepares your corpus:

```python
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
                filename = self.output_dir / f"{title.replace(' ', '_').replace('/', '_')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# {title}\n")
                    f.write(f"# Source: {page.url}\n\n")
                    f.write(page.content)

                time.sleep(1)  # Be polite to Wikipedia

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return articles

    def clean_text(self, text: str) -> str:
        """Remove Wikipedia noise and format cleanly."""
        # Remove section headers
        text = re.sub(r'=+\s*([^=]+)\s*=+', '', text)

        # Remove citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[\w+\]', '', text)

        # Remove entire sections
        sections_to_remove = [
            r'See also.*?(?=\n\n|\Z)',
            r'References.*?(?=\n\n|\Z)',
            r'External links.*?(?=\n\n|\Z)',
            r'Further reading.*?(?=\n\n|\Z)',
            r'Notes.*?(?=\n\n|\Z)',
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
            filename = self.output_dir / f"cleaned_{article['title'].replace(' ', '_').replace('/', '_')}.txt"
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
    # Define target entities
    TITLES = [
        # AI Companies
        "OpenAI", "Anthropic", "DeepMind", "Google DeepMind",
        "Google", "Microsoft", "Meta", "Tesla", "NVIDIA",
        "Hugging Face", "Stability AI", "Cohere", "Inflection AI",

        # Related Tech Companies
        "Apple", "Amazon", "IBM", "Intel", "AMD",
        "Qualcomm", "Cisco", "Oracle", "Salesforce", "Adobe",

        # Cloud/Infrastructure
        "Amazon Web Services", "Microsoft Azure", "Google Cloud",
        "Databricks", "Snowflake", "MongoDB", "Confluent",

        # People (optional - for more entities)
        "Sam Altman", "Elon Musk", "Satya Nadella", "Sundar Pichai",
        "Tim Cook", "Jensen Huang", "Demis Hassabis", "Dario Amodei",
    ]

    # Run curation
    curator = CorpusCurator(output_dir="corpus")

    print("=" * 60)
    print("CORPUS CURATION FOR GRAPHRAG LAB")
    print("=" * 60)

    print("\n1. Fetching Wikipedia articles...")
    raw_articles = curator.fetch_articles(TITLES)

    print("\n2. Cleaning and preprocessing...")
    processed_articles = curator.process_corpus(raw_articles)

    print("\n3. Generating manifest...")
    manifest = curator.generate_manifest(processed_articles)

    print("\n" + "=" * 60)
    print("CORPUS CURATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Corpus Quality Checklist

Before proceeding to entity extraction, verify your corpus:

### ✅ Minimum Requirements
- [ ] At least 50-100 documents
- [ ] Each document > 200 words after cleaning
- [ ] Documents span multiple entities (companies, people, technologies)
- [ ] Cleaned text free of Wikipedia markup/citations

### ✅ Optimal Characteristics
- [ ] Documents contain explicit relationship phrases:
  - "X founded Y"
  - "X acquired Y"
  - "X developed Y"
  - "X worked at Y"
- [ ] Entity diversity: mix of people, companies, products, technologies
- [ ] Cross-document entity overlaps (same entity appears in multiple docs)

### ❌ Common Issues to Avoid
- [ ] Stub articles (< 200 words)
- [ ] Disambiguation pages (multiple meanings)
- [ ] List articles (enumerated lists, not prose)
- [ ] Too much noise (citations, footnotes, external links)
- [ ] Single-entity focus (only about one person/company)

---

## Testing Your Corpus

Run this quick check to validate your curated corpus:

```python
from pathlib import Path
import re

def quick_corpus_check(corpus_dir: str = "corpus"):
    """Run quick quality checks on the corpus."""
    path = Path(corpus_dir)
    files = list(path.glob("cleaned_*.txt"))

    print(f"Found {len(files)} cleaned documents")

    total_words = 0
    entity_estimates = []

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        words = len(text.split())
        total_words += words

        # Rough entity estimate (capitalized words)
        entities = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        entity_estimates.append(entities)

    print(f"Total words: {total_words:,}")
    print(f"Avg words per doc: {total_words // len(files):,}")
    print(f"Avg entity mentions per doc: {sum(entity_estimates) // len(files)}")
    print(f"Total estimated entity mentions: {sum(entity_estimates):,}")

    if len(files) >= 50 and total_words > 50000:
        print("\n✓ Corpus size looks good for GraphRAG!")
    else:
        print("\n⚠ Consider adding more documents for better graph coverage")

if __name__ == "__main__":
    quick_corpus_check()
```

---

## Next Steps After Curation

Once your corpus is curated:

1. **Review the manifest** (`corpus/manifest.json`) to verify all documents are present
2. **Test with a sample**: Pick 2-3 documents and manually verify entity/relationship content
3. **Proceed to Step 1**: Entity extraction using LLM-based NER
4. **Document your corpus**: Note any quality issues or gaps for the lab report

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Wikipedia rate limiting | Add `time.sleep(2)` between requests; use batch requests |
| Disambiguation errors | Use `wikipedia.search(title)` to find correct page |
| Poor entity density | Add more documents about companies (vs. abstract concepts) |
| Articles too short | Filter for articles > 1000 words or merge related articles |
| Special characters breaking files | Use `slugify` or replace special chars in filenames |

---

## Lab Deliverables Reminder

Your corpus curation contributes to:
- **Source code**: The curation script you develop
- **Graph visualization**: The knowledge graph built from this corpus
- **Benchmark report**: Accuracy results on multi-hop questions using this corpus
- **Failure analysis**: Document any corpus quality issues that affected results

---

## Resources

- Wikipedia API: https://wikipedia.readthedocs.io/
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- LangChain text splitters: https://python.langchain.com/docs/how_to/recursive_text_splitter/
