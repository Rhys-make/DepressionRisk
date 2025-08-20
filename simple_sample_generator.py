#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单可靠的样本生成器
生成更多训练数据以提高模型性能
"""

import pandas as pd
import random

def generate_simple_training_data(target_samples=1000):
    """生成简单的训练数据"""
    
    # 抑郁相关文本（高风险）
    depression_texts = [
        "I feel so sad and hopeless today. Nothing seems to matter anymore.",
        "I'm feeling really down and depressed. Everything feels meaningless.",
        "I can't shake this feeling of sadness. It's been going on for weeks.",
        "I feel empty inside. Like there's nothing left to live for.",
        "The sadness is overwhelming. I don't know how to cope anymore.",
        "I feel so blue today. Nothing brings me joy anymore.",
        "I'm drowning in sadness. I can't see any light at the end of the tunnel.",
        "I feel worthless and hopeless. Maybe everyone would be better off without me.",
        "I'm so tired of feeling this way. I just want it to stop.",
        "I feel like I'm trapped in a dark place with no way out.",
        "Everything feels heavy and exhausting. I can't find any relief.",
        "I'm so lonely even when I'm surrounded by people.",
        "I feel like I'm a burden to everyone around me.",
        "I don't see the point in trying anymore. Nothing ever gets better.",
        "I'm so tired of pretending to be okay when I'm not.",
        "I used to love playing guitar, but now I have no interest in anything.",
        "Nothing excites me anymore. Everything feels boring and pointless.",
        "I can't find joy in the things I used to love. It's like I'm a different person.",
        "I have no motivation to do anything. Even getting out of bed is hard.",
        "I don't care about anything anymore. Everything feels meaningless.",
        "I've lost interest in all my hobbies. Nothing brings me pleasure.",
        "I can't find any reason to get up in the morning. Everything feels pointless.",
        "I used to enjoy reading, but now I can't focus on anything.",
        "I don't want to do anything. I just want to stay in bed all day.",
        "I've stopped caring about my appearance. What's the point?",
        "I don't feel like talking to anyone. I just want to be alone.",
        "I've given up on my goals. They seem impossible now.",
        "I don't see any future for myself. Everything feels hopeless.",
        "I can't sleep at night. My mind keeps racing with negative thoughts.",
        "I'm so tired all the time, but I can't seem to get any rest.",
        "I wake up in the middle of the night and can't go back to sleep.",
        "I'm exhausted but my mind won't shut off. I keep thinking about everything.",
        "I sleep too much but still feel tired. It's like I can never get enough rest.",
        "I have trouble falling asleep because my thoughts won't stop.",
        "I wake up feeling more tired than when I went to bed.",
        "I have nightmares every night. I'm afraid to go to sleep.",
        "I can't relax enough to fall asleep. My anxiety keeps me awake.",
        "I sleep for hours but never feel refreshed.",
        "I wake up early and can't go back to sleep.",
        "My sleep schedule is completely messed up.",
        "I can't concentrate on anything. My mind keeps wandering.",
        "I have trouble focusing at work. Everything seems so overwhelming.",
        "I can't seem to complete any tasks. My attention span is terrible.",
        "I keep forgetting things. It's like my brain isn't working properly.",
        "I can't make decisions anymore. Everything feels too complicated.",
        "I have trouble reading because my mind keeps drifting off.",
        "I can't remember what I was supposed to do. My memory is terrible.",
        "I feel like my brain is foggy all the time.",
        "I can't think clearly. Everything feels confused.",
        "I have trouble following conversations. My mind keeps wandering.",
        "I can't focus on my studies. I keep getting distracted.",
        "I feel like I'm getting dumber. I can't think straight.",
        "I feel like a complete failure. I can't do anything right.",
        "I'm worthless and useless. I don't deserve to be happy.",
        "I blame myself for everything that goes wrong. It's all my fault.",
        "I feel guilty about everything. I'm such a terrible person.",
        "I don't deserve good things. I'm just a burden to everyone.",
        "I'm not good enough for anything. I should just give up.",
        "I feel like I'm letting everyone down. I'm such a disappointment.",
        "I hate myself. I can't stand who I am.",
        "I feel like I'm a waste of space. I don't contribute anything.",
        "I'm so ashamed of myself. I can't face anyone.",
        "I feel like I'm a bad person. I don't deserve love.",
        "I'm constantly criticizing myself. Nothing I do is good enough.",
        "I feel like I'm a failure at life. I can't do anything right.",
        "Sometimes I think about hurting myself. Life is just too hard.",
        "I wonder if anyone would notice if I wasn't here anymore.",
        "I think about death a lot. Sometimes I wonder if life is worth living.",
        "I feel like the world would be better off without me.",
        "I have thoughts about ending it all. I just want the pain to stop.",
        "I don't see the point in living anymore. Everything is too difficult.",
        "I think about what it would be like to just disappear.",
        "I feel like I'm a burden to my family. They'd be better off without me.",
        "I have thoughts about not wanting to wake up tomorrow.",
        "I think about how peaceful it would be to just stop existing."
    ]
    
    # 正常文本（低风险）
    normal_texts = [
        "I feel so happy today! Everything is going great!",
        "Had a wonderful day with friends! We had so much fun together!",
        "I'm feeling really grateful for all the good things in my life!",
        "Today was amazing! I accomplished so much and feel proud of myself!",
        "I'm excited about the new opportunities coming my way!",
        "I feel blessed to have such wonderful people in my life!",
        "I'm in such a good mood today! Everything feels possible!",
        "I had a fantastic workout session! Feeling energized and strong!",
        "I'm feeling really optimistic about the future!",
        "I had such a peaceful and relaxing day!",
        "Just finished a challenging project at work! Feeling accomplished!",
        "I got promoted today! All my hard work is finally paying off!",
        "I completed my first marathon! Never thought I could do it!",
        "I got accepted into my dream university! Dreams do come true!",
        "I just adopted a puppy! So much love and joy in my life right now!",
        "I finished writing my book! It's been a long journey but worth it!",
        "I learned a new skill today! Always excited to grow and improve!",
        "I achieved my fitness goals! Feeling so proud of myself!",
        "I got a great job offer! My career is really taking off!",
        "I won an award for my work! Recognition feels amazing!",
        "Had an amazing dinner with family. Love spending time with them!",
        "Went to a fantastic concert last night! The music was incredible!",
        "I had a great conversation with an old friend today!",
        "I joined a new club and met some wonderful people!",
        "I went hiking with friends and the views were breathtaking!",
        "I had a fun game night with my roommates! Lots of laughter!",
        "I attended a workshop and learned so much!",
        "I reconnected with a childhood friend! It was so nice to catch up!",
        "I went to a party and had a blast! Great music and great people!",
        "I volunteered at a local charity! Helping others feels amazing!",
        "Beautiful sunset today! Nature is truly amazing and inspiring!",
        "I made a delicious meal from scratch! Cooking is so therapeutic!",
        "I read an amazing book today! It really touched my heart!",
        "I went for a peaceful walk in the park. So relaxing!",
        "I cleaned my apartment and it feels so good!",
        "I tried a new restaurant and the food was incredible!",
        "I watched a really good movie with my partner!",
        "I planted some flowers in my garden! Watching them grow is magical!",
        "I took some beautiful photos today! Capturing moments is special!",
        "I listened to my favorite music and felt so uplifted!",
        "I feel so joyful today! Everything is going great!",
        "Had a wonderful day with friends! We had so much fun together!",
        "I'm feeling really grateful for all the good things in my life!",
        "Today was amazing! I accomplished so much and feel proud of myself!",
        "I'm excited about the new opportunities coming my way!",
        "I feel blessed to have such wonderful people in my life!",
        "I'm in such a good mood today! Everything feels possible!",
        "I had a fantastic workout session! Feeling energized and strong!",
        "I'm feeling really optimistic about the future!",
        "I had such a peaceful and relaxing day!",
        "Just finished a challenging project at work! Feeling accomplished!",
        "I got promoted today! All my hard work is finally paying off!",
        "I completed my first marathon! Never thought I could do it!",
        "I got accepted into my dream university! Dreams do come true!",
        "I just adopted a puppy! So much love and joy in my life right now!",
        "I finished writing my book! It's been a long journey but worth it!",
        "I learned a new skill today! Always excited to grow and improve!",
        "I achieved my fitness goals! Feeling so proud of myself!",
        "I got a great job offer! My career is really taking off!",
        "I won an award for my work! Recognition feels amazing!",
        "Had an amazing dinner with family. Love spending time with them!",
        "Went to a fantastic concert last night! The music was incredible!",
        "I had a great conversation with an old friend today!",
        "I joined a new club and met some wonderful people!",
        "I went hiking with friends and the views were breathtaking!",
        "I had a fun game night with my roommates! Lots of laughter!",
        "I attended a workshop and learned so much!",
        "I reconnected with a childhood friend! It was so nice to catch up!",
        "I went to a party and had a blast! Great music and great people!",
        "I volunteered at a local charity! Helping others feels amazing!",
        "Beautiful sunset today! Nature is truly amazing and inspiring!",
        "I made a delicious meal from scratch! Cooking is so therapeutic!",
        "I read an amazing book today! It really touched my heart!",
        "I went for a peaceful walk in the park. So relaxing!",
        "I cleaned my apartment and it feels so good!",
        "I tried a new restaurant and the food was incredible!",
        "I watched a really good movie with my partner!",
        "I planted some flowers in my garden! Watching them grow is magical!",
        "I took some beautiful photos today! Capturing moments is special!",
        "I listened to my favorite music and felt so uplifted!"
    ]
    
    # 生成指定数量的样本
    all_texts = []
    all_labels = []
    
    # 确保数据平衡
    depression_count = target_samples // 2
    normal_count = target_samples - depression_count
    
    # 随机选择抑郁文本
    selected_depression = random.sample(depression_texts, min(depression_count, len(depression_texts)))
    if len(selected_depression) < depression_count:
        # 如果样本不够，重复使用
        while len(selected_depression) < depression_count:
            selected_depression.extend(random.sample(depression_texts, min(depression_count - len(selected_depression), len(depression_texts))))
    
    # 随机选择正常文本
    selected_normal = random.sample(normal_texts, min(normal_count, len(normal_texts)))
    if len(selected_normal) < normal_count:
        # 如果样本不够，重复使用
        while len(selected_normal) < normal_count:
            selected_normal.extend(random.sample(normal_texts, min(normal_count - len(selected_normal), len(normal_texts))))
    
    # 合并数据
    all_texts.extend(selected_depression[:depression_count])
    all_labels.extend([1] * depression_count)  # 1 = 高风险
    
    all_texts.extend(selected_normal[:normal_count])
    all_labels.extend([0] * normal_count)  # 0 = 低风险
    
    # 创建数据框
    data = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    # 打乱数据
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return data

def save_simple_data(target_samples=1000):
    """保存简单训练数据"""
    print(f"🔄 生成 {target_samples} 条训练数据...")
    
    # 生成数据
    simple_data = generate_simple_training_data(target_samples)
    
    # 保存到文件
    output_path = f"data/raw/simple_sample_data_{target_samples}.csv"
    simple_data.to_csv(output_path, index=False)
    
    print(f"✅ 简单训练数据已保存到: {output_path}")
    print(f"📊 数据统计:")
    print(f"   - 总样本数: {len(simple_data)}")
    print(f"   - 高风险样本: {len(simple_data[simple_data['label'] == 1])}")
    print(f"   - 低风险样本: {len(simple_data[simple_data['label'] == 0])}")
    print(f"   - 数据平衡性: {len(simple_data[simple_data['label'] == 1]) / len(simple_data):.2%} vs {len(simple_data[simple_data['label'] == 0]) / len(simple_data):.2%}")
    
    # 显示数据质量信息
    print(f"\n🎯 数据质量信息:")
    print(f"   - 文本长度范围: {simple_data['text'].str.len().min()}-{simple_data['text'].str.len().max()} 字符")
    print(f"   - 平均文本长度: {simple_data['text'].str.len().mean():.1f} 字符")
    print(f"   - 词汇多样性: 包含 {len(set(' '.join(simple_data['text']).split()))} 个独特词汇")
    
    return output_path

def main():
    """主函数"""
    print("🚀 简单样本生成器")
    print("=" * 50)
    
    # 用户选择样本数量
    print("请选择要生成的样本数量:")
    print("1. 500 条样本 (推荐用于快速测试)")
    print("2. 1000 条样本 (推荐用于标准训练)")
    print("3. 2000 条样本 (推荐用于高质量模型)")
    print("4. 5000 条样本 (推荐用于生产环境)")
    print("5. 自定义数量")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == "1":
        target_samples = 500
    elif choice == "2":
        target_samples = 1000
    elif choice == "3":
        target_samples = 2000
    elif choice == "4":
        target_samples = 5000
    elif choice == "5":
        try:
            target_samples = int(input("请输入自定义样本数量: "))
        except ValueError:
            print("❌ 输入无效，使用默认值 1000")
            target_samples = 1000
    else:
        print("❌ 选择无效，使用默认值 1000")
        target_samples = 1000
    
    # 生成数据
    save_simple_data(target_samples)
    
    print(f"\n🎉 成功生成 {target_samples} 条训练数据！")
    print("💡 现在可以运行 train_models.py 来训练模型")

if __name__ == "__main__":
    main()
