import os
import jieba
import pandas as pd
import numpy as np
import re
import math
from collections import Counter, defaultdict

class DocumentRecommender:
    def __init__(self, docs_folder="yuque_documents"):
        self.docs_folder = docs_folder
        self.documents = {}
        self.tokenized_docs = {}
        self.tfidf_vectors = {}
        self.vocab = set()
        
    def load_all_documents(self):
        """加载所有MD文档"""
        print("正在加载文档...")
        for root, dirs, files in os.walk(self.docs_folder):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        rel_path = os.path.relpath(file_path, self.docs_folder)
                        self.documents[rel_path] = {
                            'content': content,
                            'title': os.path.splitext(file)[0],
                            'path': rel_path,
                            'file_size': len(content),
                            'directory': os.path.dirname(rel_path)
                        }
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {e}")
        
        print(f"成功加载 {len(self.documents)} 个文档")
        return self.documents
    
    def preprocess_text(self, text):
        """文本预处理"""
        text = re.sub(r'[#*`\-\[\]\!]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_tokens(self, text):
        """提取token"""
        processed_text = self.preprocess_text(text)
        tokens = jieba.lcut(processed_text)
        tokens = [token for token in tokens if len(token) > 1 and token.strip()]
        return tokens
    
    def tokenize_all_documents(self):
        """对所有文档进行tokenize"""
        print("正在对文档进行tokenize...")
        for doc_id, doc_info in self.documents.items():
            tokens = self.extract_tokens(doc_info['content'])
            self.tokenized_docs[doc_id] = {
                'tokens': tokens,
                'token_count': len(tokens),
                'title': doc_info['title']
            }
            self.vocab.update(tokens)
        
        total_tokens = sum([info['token_count'] for info in self.tokenized_docs.values()])
        print(f"总共提取了 {total_tokens} 个token，词汇表大小: {len(self.vocab)}")
        return self.tokenized_docs
    
    def calculate_tf(self, tokens):
        """计算词频"""
        tf_dict = {}
        token_count = len(tokens)
        for token in tokens:
            tf_dict[token] = tf_dict.get(token, 0) + 1 / token_count
        return tf_dict
    
    def calculate_idf(self):
        """计算逆文档频率"""
        idf_dict = {}
        total_docs = len(self.tokenized_docs)
        
        for word in self.vocab:
            doc_count = 0
            for doc_info in self.tokenized_docs.values():
                if word in doc_info['tokens']:
                    doc_count += 1
            idf_dict[word] = math.log(total_docs / (doc_count + 1))
        return idf_dict
    
    def calculate_tfidf_vectors(self):
        """计算TF-IDF向量"""
        print("正在计算TF-IDF向量...")
        idf_dict = self.calculate_idf()
        
        for doc_id, doc_info in self.tokenized_docs.items():
            tf_dict = self.calculate_tf(doc_info['tokens'])
            tfidf_vector = {}
            
            for word in self.vocab:
                tf = tf_dict.get(word, 0)
                idf = idf_dict[word]
                tfidf_vector[word] = tf * idf
            
            self.tfidf_vectors[doc_id] = tfidf_vector
        
        print(f"已生成 {len(self.tfidf_vectors)} 个文档的TF-IDF向量")
        return self.tfidf_vectors
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        dot_product = sum(vec1[word] * vec2[word] for word in common_words)
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def jaccard_similarity(self, tokens1, tokens2):
        """计算Jaccard相似度"""
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    def calculate_combined_score(self, tfidf_score, jaccard_score, 
                               tfidf_weight=0.7, jaccard_weight=0.3):
        """计算综合评分"""
        return (tfidf_score * tfidf_weight + jaccard_score * jaccard_weight)
    
    def generate_recommendation_matrix(self):
        """生成推荐矩阵 - 核心功能"""
        print("正在生成推荐矩阵...")
        
        if not self.tfidf_vectors:
            self.calculate_tfidf_vectors()
        
        doc_ids = list(self.documents.keys())
        n_docs = len(doc_ids)
        
        # 初始化结果存储
        recommendations = []
        
        # 计算每对文档的相似度
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                doc1_id = doc_ids[i]
                doc2_id = doc_ids[j]
                
                # 计算两种相似度
                tfidf_score = self.cosine_similarity(
                    self.tfidf_vectors[doc1_id], 
                    self.tfidf_vectors[doc2_id]
                )
                
                jaccard_score = self.jaccard_similarity(
                    self.tokenized_docs[doc1_id]['tokens'],
                    self.tokenized_docs[doc2_id]['tokens']
                )
                
                # 计算综合评分
                combined_score = self.calculate_combined_score(tfidf_score, jaccard_score)
                
                # 只保留有意义的推荐
                if combined_score > 0.1:  # 阈值可根据需要调整
                    recommendations.append({
                        'source_doc': self.documents[doc1_id]['title'],
                        'source_path': doc1_id,
                        'target_doc': self.documents[doc2_id]['title'],
                        'target_path': doc2_id,
                        'tfidf_similarity': round(tfidf_score, 4),
                        'jaccard_similarity': round(jaccard_score, 4),
                        'combined_score': round(combined_score, 4),
                        'recommendation_level': self._get_recommendation_level(combined_score)
                    })
        
        # 按综合评分排序
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return recommendations
    
    def _get_recommendation_level(self, combined_score):
        """根据综合评分确定推荐级别"""
        if combined_score > 0.7:
            return "强烈推荐"
        elif combined_score > 0.5:
            return "推荐"
        elif combined_score > 0.3:
            return "一般推荐"
        else:
            return "低相关性"
    
    def generate_document_similarity_summary(self):
        """生成文档相似度摘要统计"""
        doc_ids = list(self.documents.keys())
        n_docs = len(doc_ids)
        
        # 为每个文档计算平均相似度
        doc_similarities = {}
        
        for i in range(n_docs):
            doc_id = doc_ids[i]
            similarities = []
            
            for j in range(n_docs):
                if i != j:
                    tfidf_score = self.cosine_similarity(
                        self.tfidf_vectors[doc_id], 
                        self.tfidf_vectors[doc_ids[j]]
                    )
                    jaccard_score = self.jaccard_similarity(
                        self.tokenized_docs[doc_id]['tokens'],
                        self.tokenized_docs[doc_ids[j]]['tokens']
                    )
                    combined_score = self.calculate_combined_score(tfidf_score, jaccard_score)
                    similarities.append(combined_score)
            
            if similarities:
                doc_similarities[doc_id] = {
                    'title': self.documents[doc_id]['title'],
                    'avg_similarity': round(np.mean(similarities), 4),
                    'max_similarity': round(max(similarities), 4),
                    'high_similarity_count': len([s for s in similarities if s > 0.5])
                }
        
        return doc_similarities
    
    def generate_recommendation_report(self, output_file="document_recommendations.xlsx"):
        """生成推荐报告Excel"""
        print("正在生成推荐报告...")
        
        # 获取推荐数据
        recommendations = self.generate_recommendation_matrix()
        doc_summary = self.generate_document_similarity_summary()
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: 主要推荐列表
            print("生成主要推荐列表...")
            if recommendations:
                df_recommendations = pd.DataFrame(recommendations)
                # 重新排列列顺序
                column_order = [
                    'source_doc', 'target_doc', 'combined_score', 'recommendation_level',
                    'tfidf_similarity', 'jaccard_similarity', 'source_path', 'target_path'
                ]
                df_recommendations = df_recommendations[column_order]
                df_recommendations.to_excel(writer, sheet_name='文档推荐列表', index=False)
            else:
                pd.DataFrame({'提示': ['未找到足够相似的文档对']}).to_excel(
                    writer, sheet_name='文档推荐列表', index=False)
            
            # Sheet 2: 文档相似度摘要
            print("生成文档相似度摘要...")
            if doc_summary:
                summary_data = []
                for doc_id, stats in doc_summary.items():
                    summary_data.append({
                        '文档标题': stats['title'],
                        '文档路径': doc_id,
                        '平均相似度': stats['avg_similarity'],
                        '最高相似度': stats['max_similarity'],
                        '高相似文档数': stats['high_similarity_count'],
                        '相似度等级': self._get_similarity_level(stats['avg_similarity'])
                    })
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.sort_values('平均相似度', ascending=False, inplace=True)
                df_summary.to_excel(writer, sheet_name='文档相似度摘要', index=False)
            
            # Sheet 3: 方法对比分析
            print("生成方法对比分析...")
            comparison_data = self._generate_methods_comparison(recommendations)
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='方法对比分析', index=False)
            
            # Sheet 4: 推荐统计
            print("生成推荐统计...")
            stats_data = self._generate_recommendation_stats(recommendations)
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='推荐统计', index=False)
        
        print(f"推荐报告已生成: {output_file}")
        return output_file
    
    def _get_similarity_level(self, avg_similarity):
        """根据平均相似度确定等级"""
        if avg_similarity > 0.6:
            return "高相似度文档"
        elif avg_similarity > 0.4:
            return "中等相似度文档"
        elif avg_similarity > 0.2:
            return "低相似度文档"
        else:
            return "独立文档"
    
    def _generate_methods_comparison(self, recommendations):
        """生成方法对比分析数据"""
        if not recommendations:
            return [{'指标': '无数据', '数值': '无', '说明': '未找到推荐数据'}]
        
        comparison_data = []
        
        # 提取分数
        tfidf_scores = [r['tfidf_similarity'] for r in recommendations]
        jaccard_scores = [r['jaccard_similarity'] for r in recommendations]
        combined_scores = [r['combined_score'] for r in recommendations]
        
        # 基本统计
        comparison_data.append({
            '指标': 'TF-IDF平均分', 
            '数值': f"{np.mean(tfidf_scores):.4f}", 
            '说明': '语义相似度的平均得分'
        })
        comparison_data.append({
            '指标': 'Jaccard平均分', 
            '数值': f"{np.mean(jaccard_scores):.4f}", 
            '说明': '词汇重叠度的平均得分'
        })
        comparison_data.append({
            '指标': '综合评分平均分', 
            '数值': f"{np.mean(combined_scores):.4f}", 
            '说明': '加权综合评分的平均得分'
        })
        
        # 相关性分析
        correlation = np.corrcoef(tfidf_scores, jaccard_scores)[0, 1]
        comparison_data.append({
            '指标': '方法相关性', 
            '数值': f"{correlation:.4f}", 
            '说明': 'TF-IDF和Jaccard分数的相关性'
        })
        
        # 推荐级别分布
        levels = [r['recommendation_level'] for r in recommendations]
        level_counts = Counter(levels)
        for level, count in level_counts.most_common():
            comparison_data.append({
                '指标': f'{level}数量',
                '数值': count,
                '说明': f'{level}的文档对数量'
            })
        
        return comparison_data
    
    def _generate_recommendation_stats(self, recommendations):
        """生成推荐统计数据"""
        if not recommendations:
            return [{'统计项': '无数据', '数值': '无'}]
        
        stats_data = []
        
        # 基本统计
        stats_data.append({'统计项': '总推荐对数量', '数值': len(recommendations)})
        stats_data.append({'统计项': '涉及文档数量', '数值': len(set(
            [r['source_path'] for r in recommendations] + [r['target_path'] for r in recommendations]
        ))})
        
        # 分数分布
        combined_scores = [r['combined_score'] for r in recommendations]
        stats_data.append({'统计项': '最高综合评分', '数值': f"{max(combined_scores):.4f}"})
        stats_data.append({'统计项': '最低综合评分', '数值': f"{min(combined_scores):.4f}"})
        stats_data.append({'统计项': '平均综合评分', '数值': f"{np.mean(combined_scores):.4f}"})
        
        # 推荐级别统计
        level_counts = Counter([r['recommendation_level'] for r in recommendations])
        for level, count in level_counts.most_common():
            stats_data.append({'统计项': f'{level}比例', '数值': f"{(count/len(recommendations)*100):.1f}%"})
        
        return stats_data

def main():
    """主函数"""
    print("=== 文档推荐系统 ===")
    print("基于TF-IDF和Jaccard相似度的文档推荐")
    
    # 初始化推荐器
    recommender = DocumentRecommender("yuque_documents")
    
    # 1. 加载文档
    print("步骤1: 加载文档...")
    recommender.load_all_documents()
    
    if not recommender.documents:
        print("错误: 没有找到文档，请检查yuque_documents文件夹")
        return
    
    # 2. 文本处理和分词
    print("步骤2: 文本处理和分词...")
    recommender.tokenize_all_documents()
    
    # 3. 计算TF-IDF向量
    print("步骤3: 计算文档向量...")
    recommender.calculate_tfidf_vectors()
    
    # 4. 生成推荐报告
    print("步骤4: 生成推荐报告...")
    output_file = recommender.generate_recommendation_report("文档推荐报告.xlsx")
    
    print("\n=== 推荐系统完成 ===")
    print(f"生成的文件: {output_file}")
    print("\n报告包含以下Sheet:")
    print("1. 文档推荐列表 - 主要推荐结果，按综合评分排序")
    print("2. 文档相似度摘要 - 每个文档的相似度统计")
    print("3. 方法对比分析 - 两种相似度方法的对比")
    print("4. 推荐统计 - 整体推荐数据统计")
    
    print("\n使用建议:")
    print("- 查看'文档推荐列表'获取具体的文档推荐对")
    print("- 使用'combined_score'作为主要推荐依据")
    print("- 参考'recommendation_level'确定推荐强度")

if __name__ == "__main__":
    # 检查必要依赖
    try:
        import jieba
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"缺少必要依赖: {e}")
        print("请安装: pip install jieba pandas numpy openpyxl")
        exit(1)
    
    main()
