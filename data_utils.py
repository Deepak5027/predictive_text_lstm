"""
data_utils.py — Sample text corpora and dataset helpers
"""

# ──────────────────────────────────────────────────────────────
# BUILT-IN SAMPLE CORPORA
# ──────────────────────────────────────────────────────────────

SAMPLE_CORPORA = {
    "📰 Tech & AI News": """
Artificial intelligence is transforming the way we live and work every single day. Machine
learning algorithms analyze massive datasets to uncover patterns that humans cannot detect.
Deep learning neural networks have achieved remarkable accuracy in image recognition tasks.
Natural language processing enables computers to understand and generate human language.
The rise of large language models has opened new possibilities for human computer interaction.
Researchers continue to push the boundaries of what artificial intelligence can achieve.
Autonomous vehicles use machine learning to navigate complex real world environments safely.
Recommendation systems powered by deep learning personalize content for millions of users.
The future of technology depends on responsible development of artificial intelligence systems.
Data scientists train models on labeled datasets to improve prediction accuracy over time.
Transfer learning allows models to apply knowledge from one domain to another efficiently.
Reinforcement learning teaches agents to make optimal decisions through trial and error.
Computer vision systems can now identify objects in images with superhuman precision.
Speech recognition technology has made voice interfaces a standard part of modern devices.
The ethics of artificial intelligence requires careful consideration of bias and fairness.
Edge computing brings machine learning inference closer to the data source for faster results.
Generative models can create realistic images, audio, and text from simple prompts.
Federated learning allows models to train on distributed data without centralizing privacy.
Explainability in machine learning helps users understand why a model made a specific decision.
The intersection of neuroscience and artificial intelligence continues to inspire new architectures.
""",

    "📚 Classic Literature": """
It was the best of times it was the worst of times it was the age of wisdom it was the age of
foolishness. The sun also rises and the rivers run into the sea yet the sea is never full.
Call me Ishmael some years ago never mind how long precisely having little money in my pocket.
To be or not to be that is the question whether tis nobler in the mind to suffer the slings.
It is a truth universally acknowledged that a single man in possession of a good fortune must be.
All happy families are alike each unhappy family is unhappy in its own way and life goes on.
In the beginning was the word and the word was with meaning and the word carried all truth.
The only way out of the labyrinth of suffering is to forgive and to love without condition.
We are all fools in love and love itself is the greatest fool that ever walked this earth.
Time passes and memories fade but the written word lives on beyond the lives of those who wrote.
The great novel captures something true about the human condition beyond what facts can express.
Character is destiny and destiny is shaped by the choices we make in our darkest moments.
The hero of the story is not always the strongest but the one who endures through suffering.
Words have power beyond their meaning and stories carry truths that facts alone cannot hold.
Every great story is ultimately about love loss courage and the search for meaning in life.
The narrator looked back on the years and wondered what might have been different had he chosen.
She stood at the window watching the rain fall on the empty street below thinking of the past.
The old man had lived long enough to know that happiness was not a destination but a practice.
Between the lines of every great book lies the unspoken truth the author could not say aloud.
Literature is the conversation humanity has with itself across centuries of time and change.
""",

    "🧬 Science & Nature": """
The universe began with a massive expansion of energy known as the big bang approximately fourteen
billion years ago. Stars form from clouds of gas and dust collapsing under the force of gravity.
Black holes are regions of spacetime where gravity is so strong that nothing can escape their pull.
The theory of evolution explains how life on earth has changed over billions of years of time.
DNA carries the genetic instructions for the development and function of all living organisms.
Photosynthesis is the process by which plants convert sunlight into chemical energy for growth.
The periodic table organizes all known chemical elements by their atomic number and properties.
Climate change is driven by the accumulation of greenhouse gases in the earths atmosphere today.
Ocean currents regulate the temperature of the planet and influence weather patterns globally.
Quantum mechanics describes the behavior of particles at the smallest scales of the universe.
Neurons communicate through electrical and chemical signals to produce thought and consciousness.
The immune system defends the body against pathogens using a complex network of cells and proteins.
Plate tectonics explains the movement of the earths crust and the formation of mountains and ocean.
Biodiversity is essential for the resilience of ecosystems and the services they provide to life.
The water cycle continuously moves water through the atmosphere oceans and land in a closed loop.
Genetic mutations drive evolution by creating variation in populations over many generations.
The speed of light is the universal speed limit set by the laws of physics in our universe.
Ecosystems are complex networks of organisms interacting with each other and their environment.
Scientific inquiry relies on observation hypothesis testing and peer review to advance knowledge.
The human brain contains approximately eighty six billion neurons forming trillions of connections.
""",

    "💬 Conversational & Chat": """
How are you doing today I hope everything is going well for you and your family.
The weather has been really nice lately and I have been enjoying my morning walks outside.
I was thinking we could meet up this weekend if you are free and have some time available.
Did you hear about the new restaurant that opened downtown I heard the food is absolutely amazing.
I have been working on a new project at work and it is taking up most of my time these days.
Let me know if you need any help with anything I am always happy to lend a hand when needed.
Have you watched any good movies lately I am looking for something interesting to watch tonight.
The meeting tomorrow has been rescheduled to the afternoon so please update your calendar.
I really enjoyed our conversation last time and I hope we can catch up again very soon.
Thank you so much for all your help it really made a huge difference in how things turned out.
Can you send me the report when you have a chance I need to review it before the meeting.
It has been a while since we last spoke I hope you have been keeping well during this time.
I was wondering if you had any recommendations for good books to read over the coming weeks.
The team did an amazing job on the presentation everyone was really impressed by the results.
I think we should discuss this further before making any final decisions on the matter.
Please feel free to reach out if you have any questions or need any clarification on anything.
Looking forward to hearing from you and hope we can connect again very soon this month.
It would be great to collaborate on something together I think we could do great work as a team.
Let me check my schedule and get back to you with some times that might work for both of us.
I appreciate your patience and understanding throughout this process it means a lot to me.
""",
}


def get_corpus_names():
    return list(SAMPLE_CORPORA.keys())


def get_corpus_text(name: str) -> str:
    return SAMPLE_CORPORA.get(name, "").strip()


def get_corpus_stats(text: str) -> dict:
    import re
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    unique = set(words)
    return {
        "total_words":    len(words),
        "unique_words":   len(unique),
        "sentences":      len(sentences),
        "avg_word_len":   round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "lexical_density": round(len(unique) / max(len(words), 1) * 100, 1),
    }


def get_top_words(text: str, n: int = 20) -> list:
    """Return top-n most frequent words, excluding stopwords."""
    from collections import Counter
    import re

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "it", "its", "this", "that",
        "these", "those", "i", "we", "you", "he", "she", "they", "my", "our",
        "your", "his", "her", "their", "not", "no", "so", "as", "if", "up",
        "out", "about", "all", "what", "which", "who", "how", "when", "where",
    }
    words = re.findall(r"[a-z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(n)
