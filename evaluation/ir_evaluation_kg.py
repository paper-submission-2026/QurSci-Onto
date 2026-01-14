"""
IR Evaluation: Knowledge Graph Enhanced Retrieval
=================================================
Tests the performance of KG-enriched embeddings against baseline.

Key Features:
- Uses hasScientificConceptID and hasScientificNodes from ayah_ontology_latest_with_arabic.csv
- Enriches with hasScientificKeywords + hasTafsirSummary from scientific_ontology_latest.csv
- Tests on scientific queries targeting annotated verses
- Compares baseline (plain text) vs KG-enhanced embeddings
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
import ast
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from scipy import stats


class KnowledgeGraphIREvaluator:
    """IR evaluation with Knowledge Graph enrichment"""
    
    def __init__(self):
        print("🔬 Knowledge Graph Enhanced IR Evaluation")
        print("="*70)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.baseline_vectorstore = None
        self.enhanced_vectorstore = None
        
        # Load datasets
        self.ayah_df = pd.read_csv('data/ayah_ontology.csv')
        self.exegesis_df = pd.read_csv('data/exegesis_rows.csv')
        self.scientific_df = pd.read_csv('data/scientific_ontology.csv')
        self.quran_df = pd.read_csv('data/The Quran Dataset.csv')
        
        print(f"✓ Loaded Ayah Ontology: {len(self.ayah_df)} verses")
        print(f"✓ Loaded Scientific KG: {len(self.scientific_df)} nodes")
        print(f"✓ Loaded Quran Dataset: {len(self.quran_df)} verses")
        
        # Add after loading datasets
        print("\n🔍 Checking column names in scientific_df:")
        print("Available columns:", list(self.scientific_df.columns))

        # Check for required columns
        required_cols = ['hasNodeID', 'hasScientificKeywords', 'hasTafsirSummary']
        for col in required_cols:
            if col in self.scientific_df.columns:
                print(f"✓ Found: {col}")
            else:
                print(f"❌ MISSING: {col}")
                
        # Check what columns actually exist with "Keywords" in name
        keyword_cols = [col for col in self.scientific_df.columns if 'Keyword' in col]
        print(f"\nKeyword-related columns: {keyword_cols}")
        
        # Build knowledge graph lookup
        self.kg_lookup = self._build_kg_lookup()
        
    def parse_list_field(self, value):
        """Safely parse list fields"""
        if pd.isna(value):
            return []
        if isinstance(value, str) and value.startswith('['):
            try:
                return ast.literal_eval(value)
            except:
                return []
        return []
    
    def _build_kg_lookup(self):
        """Build lookup: (surah, ayah) -> KG enrichment data"""
        print("\n📚 Building Knowledge Graph Lookup...")
        
        kg_lookup = {}
        
        # Build node lookup from scientific_df
        node_data = {}
        for _, row in self.scientific_df.iterrows():
            node_id = row['hasNodeID']
            node_data[node_id] = {
                'topic': row['hasTopicID'],
                'description': row['hasScientificKeywords'],
                'tafsir': row['hasTafsirSummary'],
                'quranic_term': row['hasQuranicTermArabic'],
                'type': row.get('hasType', ''),
                'relation': row['hasRelation']
            }
        
        # Map verses to their KG nodes
        for _, row in self.ayah_df.iterrows():
            key = (row['surah_no'], row['ayah_no'])
            
            enrichment = {
                'topics': self.parse_list_field(row['hasScientificTopics']),
                'concepts': self.parse_list_field(row['hasScientificConceptID']),
                'nodes': self.parse_list_field(row['hasScientificNodes']),
                'categories': self.parse_list_field(row['hasBroadCategories']),
                'themes': self.parse_list_field(row['hasThemes']),
                'kg_data': []
            }
            
            # Get detailed KG data for linked nodes
            for node_id in enrichment['nodes']:
                if node_id in node_data:
                    enrichment['kg_data'].append(node_data[node_id])
            
            kg_lookup[key] = enrichment
        
        print(f"✓ Built KG lookup for {len(kg_lookup)} annotated verses")
        print(f"  Verses with concept links: {sum(1 for v in kg_lookup.values() if v['concepts'])}")
        print(f"  Verses with node links: {sum(1 for v in kg_lookup.values() if v['nodes'])}")
        
        return kg_lookup
    
    def create_baseline_vectorstore(self):
        """Create baseline vectorstore with plain Quran text"""
        print("\n" + "="*70)
        print("STEP 1: Creating BASELINE Vectorstore (Plain Text)")
        print("="*70)
        
        # Get all annotated verses
        annotated_keys = set(self.kg_lookup.keys())
        
        # Filter Quran dataset to annotated verses
        quran_annotated = self.quran_df[
            self.quran_df.apply(
                lambda x: (x['surah_no'], x['ayah_no_surah']) in annotated_keys,
                axis=1
            )
        ].copy()
        
        print(f"  Indexing {len(quran_annotated)} annotated verses")
        
        documents = []
        for _, row in quran_annotated.iterrows():
            text = f"{row['surah_name_en']} {row['ayah_no_surah']}: {row['ayah_en']}"
            
            doc = Document(
                page_content=text,
                metadata={
                    'surah_no': int(row['surah_no']),
                    'ayah_no': int(row['ayah_no_surah']),
                    'surah_name': row['surah_name_en'],
                    'type': 'baseline'
                }
            )
            documents.append(doc)
        
        print("  Creating embeddings...")
        self.baseline_vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"✓ Baseline vectorstore: {len(documents)} documents")
        
    def create_enhanced_vectorstore(self):
        """Create KG-enhanced vectorstore"""
        print("\n" + "="*70)
        print("STEP 2: Creating KG-ENHANCED Vectorstore")
        print("="*70)
        
        # Get all annotated verses
        annotated_keys = set(self.kg_lookup.keys())
        
        # Filter Quran dataset to annotated verses
        quran_annotated = self.quran_df[
            self.quran_df.apply(
                lambda x: (x['surah_no'], x['ayah_no_surah']) in annotated_keys,
                axis=1
            )
        ].copy()
        
        documents = []
        kg_enriched_count = 0
        
        for _, row in quran_annotated.iterrows():
            key = (row['surah_no'], row['ayah_no_surah'])
            
            # Base text
            enhanced_text = f"{row['surah_name_en']} {row['ayah_no_surah']}: {row['ayah_en']}"
            
            enrichments = []
            metadata = {
                'surah_no': int(row['surah_no']),
                'ayah_no': int(row['ayah_no_surah']),
                'surah_name': row['surah_name_en'],
                'type': 'enhanced',
                'has_kg': False
            }
            
            # Get KG enrichment
            if key in self.kg_lookup:
                kg_data = self.kg_lookup[key]
                
                # Add concept IDs
                if kg_data['concepts']:
                    enrichments.append(f"Concepts: {', '.join(kg_data['concepts'])}")
                
                # Add detailed KG node information
                if kg_data['kg_data']:
                    kg_enriched_count += 1
                    metadata['has_kg'] = True
                    
                    # Collect scientific descriptions
                    descriptions = [node['description'] for node in kg_data['kg_data'] 
                                   if pd.notna(node['description'])]
                    if descriptions:
                        enrichments.append(f"Scientific: {' | '.join(descriptions[:3])}")
                    
                    # Collect tafsir summaries
                    tafsir = [node['tafsir'] for node in kg_data['kg_data'] 
                             if pd.notna(node['tafsir'])]
                    if tafsir:
                        enrichments.append(f"Tafsir: {' | '.join(tafsir[:2])}")
                    
                    # Add Quranic terms
                    terms = [node['quranic_term'] for node in kg_data['kg_data'] 
                            if pd.notna(node['quranic_term']) and node['quranic_term'] != 'Tafsir-Derived']
                    if terms:
                        enrichments.append(f"Arabic Terms: {', '.join(terms[:3])}")
                
                # Add topics as fallback
                elif kg_data['topics']:
                    enrichments.append(f"Topics: {', '.join(kg_data['topics'])}")
            
            # Build enhanced text
            if enrichments:
                enhanced_text += " | " + " | ".join(enrichments)
            
            doc = Document(page_content=enhanced_text, metadata=metadata)
            documents.append(doc)
        
        print("  Creating embeddings...")
        self.enhanced_vectorstore = FAISS.from_documents(documents, self.embeddings)
        print(f"✓ Enhanced vectorstore: {len(documents)} documents")
        print(f"  - KG enriched: {kg_enriched_count}")
        print(f"  - Coverage: {kg_enriched_count/len(documents)*100:.1f}%")
        
    def define_test_queries(self) -> List[Dict]:
        """Define focused test queries targeting KG-annotated verses in current dataset"""
    
        
        # OLD QUERIES (Commented out - not all verses in current dataset)
        # # COSMOLOGY - BigBang_COSMOS_EXPANSION
        return [
        {
            'query': 'big bang universe creation heavens earth joined separated ratq fatq',
            'ground_truth': [(21, 30)],
            'category': 'cosmology'
        },
        {
            'query': 'cosmic smoke gaseous nebula early universe dukhan',
            'ground_truth': [(41, 11)],
            'category': 'cosmology'
        },
        {
            'query': 'expanding universe space heavens continuous expansion musioon',
            'ground_truth': [(51, 47)],
            'category': 'cosmology'
        },
        {
            'query': 'universe collapse big crunch heavens rolled scroll',
            'ground_truth': [(21, 104)],
            'category': 'cosmology'
        },
        
        # OCEANOGRAPHY - ESTUARY
        {
            'query': 'two seas barrier barzakh salt fresh water mixing estuarine',
            'ground_truth': [(25, 53), (55, 19), (55, 20)],
            'category': 'oceanography'
        },
        {
            'query': 'fresh water salt water meet partition separation',
            'ground_truth': [(55, 20), (25, 53)],
            'category': 'oceanography'
        },
        {
            'query': 'pycnocline density stratification water bodies',
            'ground_truth': [(55, 20)],
            'category': 'oceanography'
        },
        
        # HYDROLOGY - HYDRO_METEO
        {
            'query': 'water cycle clouds rain formation precipitation',
            'ground_truth': [(24, 43), (30, 48)],
            'category': 'hydrology'
        },
        {
            'query': 'clouds mountains hail lightning thunder cumulonimbus',
            'ground_truth': [(24, 43)],
            'category': 'hydrology'
        },
        {
            'query': 'wind driving clouds advection convergence',
            'ground_truth': [(24, 43), (15, 22)],
            'category': 'hydrology'
        },
        {
            'query': 'rain infiltration groundwater aquifer recharge',
            'ground_truth': [(23, 18), (39, 21)],
            'category': 'hydrology'
        },
        
        # EMBRYOLOGY - EMBRYO
        {
            'query': 'embryo development stages nutfah alaqah mudghah sequential',
            'ground_truth': [(23, 12), (23, 13), (23, 14)],
            'category': 'embryology'
        },
        {
            'query': 'sperm drop nutfah gamete fertilization zygote',
            'ground_truth': [(76, 2), (23, 13)],
            'category': 'embryology'
        },
        {
            'query': 'clinging clot leech alaqah blastocyst implantation',
            'ground_truth': [(96, 2), (23, 14)],
            'category': 'embryology'
        },
        {
            'query': 'bones covered flesh muscle ossification myogenesis',
            'ground_truth': [(23, 14)],
            'category': 'embryology'
        },
        {
            'query': 'three darkness layers womb anatomical membranes',
            'ground_truth': [(39, 6)],
            'category': 'embryology'
        },
        
        # BIOLOGY - BEE_BIO
        {
            'query': 'honey bees instinct foraging nectar collection',
            'ground_truth': [(16, 68), (16, 69)],
            'category': 'biology'
        },
        {
            'query': 'bee products healing medicinal antibacterial',
            'ground_truth': [(16, 69)],
            'category': 'biology'
        },
        {
            'query': 'bee hive hexagonal architecture wax structure',
            'ground_truth': [(16, 68)],
            'category': 'biology'
        },
        
        # GENERAL QUERIES
        {
            'query': 'scientific miracles cosmology universe creation',
            'ground_truth': [(21, 30), (41, 11), (51, 47)],
            'category': 'general'
        },
        {
            'query': 'water phenomena oceans seas barriers mixing',
            'ground_truth': [(25, 53), (55, 19), (55, 20)],
            'category': 'general'
        },
        {
            'query': 'human embryonic development fetal stages',
            'ground_truth': [(23, 12), (23, 13), (23, 14), (96, 2)],
            'category': 'general'
        },
        {
            'query': 'atmospheric processes weather clouds precipitation',
            'ground_truth': [(24, 43), (30, 48), (15, 22)],
            'category': 'general'
        },
        {
            'query': 'animal behavior biological systems nature',
            'ground_truth': [(16, 68), (16, 69), (29, 41)],
            'category': 'general'
        },
        ]
    
    def evaluate_query(self, query: str, ground_truth: List[Tuple[int, int]], 
                       vectorstore, k: int = 10) -> Dict:
        """Evaluate single query"""
        results = vectorstore.similarity_search(query, k=k)
        
        retrieved = [(doc.metadata['surah_no'], doc.metadata['ayah_no']) 
                     for doc in results]
        
        metrics = {}
        
        # Precision@K and Recall@K
        for k_val in [1, 3, 5, 10]:
            retrieved_k = retrieved[:k_val]
            relevant_retrieved = len([v for v in retrieved_k if v in ground_truth])
            metrics[f'P@{k_val}'] = relevant_retrieved / k_val if k_val > 0 else 0
            metrics[f'R@{k_val}'] = relevant_retrieved / len(ground_truth) if ground_truth else 0
        
        # MRR
        first_relevant_rank = None
        for rank, verse in enumerate(retrieved, 1):
            if verse in ground_truth:
                first_relevant_rank = rank
                break
        metrics['MRR'] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        # NDCG@10
        dcg = 0.0
        for rank, verse in enumerate(retrieved[:10], 1):
            if verse in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
        
        idcg = sum(1.0 / np.log2(rank + 1) 
                   for rank in range(1, min(len(ground_truth), 10) + 1))
        metrics['NDCG@10'] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n" + "="*70)
        print("STEP 3: Running IR Evaluation")
        print("="*70)
        
        test_queries = self.define_test_queries()
        
        baseline_results = []
        enhanced_results = []
        
        print(f"\n🔍 Evaluating {len(test_queries)} queries...\n")
        
        for i, q in enumerate(test_queries, 1):
            print(f"  [{i}/{len(test_queries)}] {q['query'][:55]}...")
            
            baseline_metrics = self.evaluate_query(
                q['query'], q['ground_truth'], 
                self.baseline_vectorstore
            )
            baseline_metrics['query'] = q['query']
            baseline_metrics['category'] = q['category']
            baseline_results.append(baseline_metrics)
            
            enhanced_metrics = self.evaluate_query(
                q['query'], q['ground_truth'],
                self.enhanced_vectorstore
            )
            enhanced_metrics['query'] = q['query']
            enhanced_metrics['category'] = q['category']
            enhanced_results.append(enhanced_metrics)
        
        return baseline_results, enhanced_results
    
    def aggregate_and_report(self, baseline_results: List[Dict], 
                            enhanced_results: List[Dict]):
        """Generate comprehensive report with statistical testing"""
        print("\n" + "="*70)
        print("STEP 4: Statistical Analysis & Reporting")
        print("="*70)
        
        baseline_df = pd.DataFrame(baseline_results)
        enhanced_df = pd.DataFrame(enhanced_results)
        
        # Aggregate metrics
        summary_rows = []
        metric_cols = ['P@1', 'P@3', 'P@5', 'P@10', 'R@5', 'MRR', 'NDCG@10']
        
        print("\n📊 AGGREGATE RESULTS:")
        print("-" * 70)
        
        for metric in metric_cols:
            b_vals = baseline_df[metric].values
            e_vals = enhanced_df[metric].values
            
            b_mean = b_vals.mean()
            e_mean = e_vals.mean()
            delta = e_mean - b_mean
            delta_pct = (delta / b_mean * 100) if b_mean > 0 else 0
            
            # Paired t-test
            _, p_value = stats.ttest_rel(e_vals, b_vals)
            
            summary_rows.append({
                'Metric': metric,
                'Baseline': round(b_mean, 4),
                'Enhanced': round(e_mean, 4),
                'Delta_Absolute': round(delta, 4),
                'Delta_Percent': round(delta_pct, 2),
                'p_value': round(p_value, 4),
                'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            })
            
            sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            print(f"{metric:10} | B: {b_mean:.4f} | E: {e_mean:.4f} | "
                  f"Δ: +{delta:.4f} (+{delta_pct:6.2f}%) | p={p_value:.4f} {sig_marker}")
        
        # Per-category analysis
        print("\n📈 PER-CATEGORY PERFORMANCE:")
        print("-" * 70)
        
        category_summary_rows = []
        for category in sorted(baseline_df['category'].unique()):
            b_cat = baseline_df[baseline_df['category'] == category]
            e_cat = enhanced_df[enhanced_df['category'] == category]
            
            # Calculate NDCG@10 for this category
            b_ndcg = b_cat['NDCG@10'].mean()
            e_ndcg = e_cat['NDCG@10'].mean()
            improvement = ((e_ndcg - b_ndcg) / b_ndcg * 100) if b_ndcg > 0 else 0
            
            category_summary_rows.append({
                'Category': category.capitalize(),
                'Queries': len(b_cat),
                'Baseline_NDCG': round(b_ndcg, 3),
                'Enhanced_NDCG': round(e_ndcg, 3),
                'Improvement': f"+{improvement:.1f}%"
            })
            
            print(f"\n{category.upper()} ({len(b_cat)} queries):")
            for metric in ['P@5', 'MRR', 'NDCG@10']:
                b_mean = b_cat[metric].mean()
                e_mean = e_cat[metric].mean()
                delta = e_mean - b_mean
                print(f"  {metric}: {b_mean:.3f} → {e_mean:.3f} (Δ: +{delta:.3f})")
        
        # Save category-wise summary
        category_summary_df = pd.DataFrame(category_summary_rows)
        os.makedirs('results', exist_ok=True)
        category_summary_df.to_csv('results/ir_evaluation_category_summary.csv', index=False)
        print(f"\n✓ Category summary saved to: results/ir_evaluation_category_summary.csv")
        
        # Improvement consistency
        print("\n📌 IMPROVEMENT CONSISTENCY:")
        print("-" * 70)
        
        for metric in ['P@5', 'MRR']:
            deltas = (enhanced_df[metric] - baseline_df[metric]).values
            improved = (deltas > 0).sum()
            worse = (deltas < 0).sum()
            same = (deltas == 0).sum()
            
            print(f"\n{metric}:")
            print(f"  Improved: {improved}/{len(deltas)} ({improved/len(deltas)*100:.1f}%)")
            print(f"  Worse: {worse}/{len(deltas)} ({worse/len(deltas)*100:.1f}%)")
            print(f"  Same: {same}/{len(deltas)} ({same/len(deltas)*100:.1f}%)")
        
        # Save results
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('results/ir_evaluation_summary.csv', index=False)
        
        results_json = {
            'metadata': {
                'evaluation_type': 'knowledge_graph_enrichment',
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(baseline_results),
                'annotated_verses': len(self.kg_lookup),
                'kg_nodes': len(self.scientific_df)
            },
            'baseline_per_query': baseline_results,
            'enhanced_per_query': enhanced_results,
            'aggregate_summary': summary_rows
        }
        
        with open('results/ir_evaluation_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print("\n✅ RESULTS SAVED:")
        print("  - results/ir_evaluation_summary.csv")
        print("  - results/ir_evaluation_category_summary.csv")
        print("  - results/ir_evaluation_results.json")
        
        return summary_df


def main():
    """Main execution"""
    print("╔" + "="*68 + "╗")
    print("║" + " IR EVALUATION: KNOWLEDGE GRAPH ENHANCED RETRIEVAL ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    evaluator = KnowledgeGraphIREvaluator()
    
    # Create vectorstores
    evaluator.create_baseline_vectorstore()
    evaluator.create_enhanced_vectorstore()
    
    # Run evaluation
    baseline_results, enhanced_results = evaluator.run_evaluation()
    
    # Generate report
    summary_df = evaluator.aggregate_and_report(baseline_results, enhanced_results)
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("  ✓ Tested KG-enriched vs baseline embeddings")
    print("  ✓ Used hasScientificKeywords + hasTafsirSummary from NEW FIXED KG")
    print("  ✓ Evaluated on 24 comprehensive queries")
    print("  ✓ Statistical significance testing applied")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
