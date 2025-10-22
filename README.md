<div align="center">
  <h1>
    <img src="assets/logo.png" width="200" alt="IF-VidCap Logo" style="vertical-align: middle; margin-bottom: 10px;"><br>
    IF-VidCap:<br>
    Can Video Caption Models Follow Instructions?
  </h1>
  
  <p align="center">
    <a href="https://github.com/NJU-LINK/IF-VidCap"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://arxiv.org/abs/2510.18726"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://if-vidcap.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/NJU-LINK/IF-VidCap"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow?style=for-the-badge" alt="Dataset"></a>
  </p>

  <p align="center">
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
  </p>
</div>

---

## 📋 Abstract

Although Multimodal Large Language Models (MLLMs) have demonstrated proficiency in video captioning, practical applications require captions that follow specific user instructions rather than generating exhaustive, unconstrained descriptions. Current benchmarks, however, primarily assess descriptive comprehensiveness while largely overlook instruction-following capabilities. 

To address this gap, we introduce **IF-VidCap**, a new benchmark for evaluating controllable video captioning, which contains 1,400 high-quality samples. Distinct from existing video captioning or general instruction-following benchmarks, IF-VidCap incorporates a systematic framework that assesses captions on two dimensions: **format correctness** and **content correctness**.

<p align="center">
  <img src="assets/first_page.png" width="800" alt="IF-VidCap Overview">
  <br>
  <em>Figure 1: Differences in Controlled Video Captioning Capabilities among MLLMs</em>
</p>

## 🌟 Key Features

- **🎯 First Instruction-Following Video Captioning Benchmark**: 1,400 complex, compositional instructions aligned with real-world downstream applications
- **🔍 Robust Evaluation Protocol**: Multi-dimensional evaluation combining rule-based and LLM-based checks
- **📊 Comprehensive Analysis**: Evaluation of 20+ state-of-the-art models with detailed insights
- **📚 Training Dataset**: Curated dataset for fine-grained instruction-based control

<p align="center">
  <img src="assets/case.png" width="800" alt="IF-VidCap Overview">
  <br>
  <em>Figure 2: Sample data in IF-VidCap. Our checklist is divided into two types based on the checking method: rule-based items checked by LLM with rule scripts and open-ended items checked by LLM. The rule-based items cover format correctness, while the open-ended items cover semantic and content correctness.</em>
</p>

## 📈 Benchmark Statistics

<p align="center">
  <img src="assets/statistics.png" width="800" alt="Dataset Statistics">
</p>

- **Video Duration**: Average 20.5s (ranging from 3s to 60s)
- **Constraint Types**: 27 distinct types across 6 categories
- **Average Constraints**: 6 per instruction
- **Video Categories**: 13+ diverse categories including Film & TV, Animation, Sports, Nature, etc.

## 📰 News
- **[22/10/2025]** 📝 Our paper is now available on arXiv
- **[22/10/2025]** 🤗 Dataset is now available on Hugging Face
- **[Coming Soon]** 🚀 Evaluation scripts will be available soon
- **[Coming Soon]** 🚀 Training dataset and code will be released

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/NJU-LINK/IF-VidCap.git
cd IF-VidCap
pip install openai
```

### Download Dataset

```bash
# use huggingface-cli
hf download NJU-LINK/IF-VidCap --local-dir ./IF-VidCap --include-pattern "*.mp4"
```

### Evaluation

```python
python generate_check_result.py -w 30 -m example
```

## 📂 File Structure

```
IF-VidCap/
├── videos/     # Video files
│   ├── clip/           
│   ├── short/
├── annotation/   # Annotations
│   ├── checklist.json
│   ├── prompt.json
│   └── video_meta_info.json
├── meta_prompt/
│   ├── open_ended_judge_llm_meta_prompt.txt
│   ├── rule_based_judge_llm_meta_prompt.txt
│   └── test_vlm_meta_prompt.txt
├── models/     # Models to be tested 
├── utils/
├── inference/
│   ├── get_response_qwen.py       # Inference script for Qwen-based models
│   ...
├── response/     # Model responses to be tested, naming convention: {model_name}_response.json
├── generate_check_result.py      # Script to generate check results by LLM
├── metrics.py                    # Script to compute metrics
```

### Dataset Card

Visit our [Hugging Face Dataset Page](https://huggingface.co/datasets/NJU-LINK/IF-VidCap) for:
- 📊 Detailed dataset statistics
- 📝 Data format specifications
- 🔍 Example viewer
- 📄 License information

## 📊 Benchmark Results

### Overall Performance

| Model | Params | Overall ISR | Overall CSR | Rule-based ISR | Rule-based CSR | Open-ended ISR | Open-ended CSR |
|-------|--------|-------------|-------------|----------------|----------------|----------------|----------------|
| **Closed-Source Models** |
| Gemini-2.5-Pro | - | 27.83 | 74.53 | 74.35 | 87.81 | 35.22 | 59.00 |
| GPT-4o | - | 22.90 | 70.74 | 69.20 | 85.12 | 30.94 | 53.91 |
| **Open-Source Models** |
| Qwen3-VL-72B | 72B | 26.41 | 71.65 | 67.16 | 84.14 | 36.39 | 57.12 |
| InternVL-3.5 | 241B | 24.20 | 71.17 | 65.58 | 83.21 | 34.64 | 57.13 |
| Qwen2.5-VL-32B | 32B | 15.16 | 64.04 | 53.66 | 76.95 | 26.72 | 48.94 |
| **IF-Captioner-Qwen (Ours)** | 7B | **12.76** | **61.64** | **58.50** | **78.81** | **19.65** | **41.56** |

*ISR: Instruction Satisfaction Rate, CSR: Constraint Satisfaction Rate*

### Key Findings

1. 📈 **Performance scales with model size** within the same family
2. 🏆 **Top open-source models now rival closed-source** counterparts
3. 🧠 **Reasoning capabilities are crucial** for complex instruction-following
4. 📝 **Format control is easier than content control** across all models

## 🛠️ Training Your Own Model
### 🚧 Training Dataset (Coming Soon)
We are preparing to release our training dataset on Hugging Face. The dataset contains:
- 11K curated video-caption pairs
- 46K video-instruction-response triplets
- Diverse instruction types covering all 27 constraint categories
**Expected release date**: Coming soon! Follow our [Hugging Face page](https://huggingface.co/datasets/NJU-LINK/IF-VidCap) for updates.



## 📝 Citation

If you find our work useful, please cite:

```bibtex
@misc{li2025ifvidcapvideocaptionmodels,
      title={IF-VidCap: Can Video Caption Models Follow Instructions?}, 
      author={Shihao Li and Yuanxing Zhang and Jiangtao Wu and Zhide Lei and Yiwen He and Runzhe Wen and Chenxi Liao and Chengkang Jiang and An Ping and Shuo Gao and Suhan Wang and Zhaozhou Bian and Zijun Zhou and Jingyi Xie and Jiayi Zhou and Jing Wang and Yifan Yao and Weihao Xie and Yingshui Tan and Yanghai Wang and Qianqian Xie and Zhaoxiang Zhang and Jiaheng Liu},
      year={2025},
      eprint={2510.18726},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.18726}, 
}
```

## 📄 License

Our dataset is under the CC-BY-NC-SA-4.0 license.

## 📧 Contact

For questions and feedback:
- 🐛 Issues: [GitHub Issues](https://github.com/NJU-LINK/IF-VidCap/issues)
- 💬 Discussions: [Hugging Face Discussions](https://huggingface.co/datasets/NJU-LINK/IF-VidCap/discussions)
- 📧 Email: [lishihao@smail.nju.edu.cn](mailto:lishihao@smail.nju.edu.cn)

