package ma.emsi.khairat.test1_ragnaif;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;                 // ✅ manquant
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaif {

    private static void configureLogger() {
        System.out.println("Configuring logger");
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger(); // ✅ Active le logging détaillé


        System.out.println("=== Phase 1 : Enregistrement des embeddings ===");

        // 1️⃣ Création du parser PDF (Apache Tika)
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        // 2️⃣ Chargement du fichier PDF
        Path path = Paths.get("src/main/resources/rag-2.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);

        // 3️⃣ Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments : " + segments.size());

        // 4️⃣ Création du modèle d’embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 5️⃣ Génération des embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre d'embeddings générés : " + embeddings.size());

        // 6️⃣ Création du magasin d’embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // 7️⃣ Ajout des embeddings et segments associés
        embeddingStore.addAll(embeddings, segments);

        System.out.println("✅ Enregistrement des embeddings terminé avec succès !");

        System.out.println("\n=== Phase 2 : Recherche et réponse avec Gemini ===");

        // 🔑 Ta clé Gemini
        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("❌ Variable d'environnement GEMINI_KEY manquante !");
        }

        // 🧠 1️⃣ Création du modèle de chat Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        // 📚 2️⃣ Création du ContentRetriever
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 💬 3️⃣ Ajout d'une mémoire de 10 messages
        var memory = MessageWindowChatMemory.withMaxMessages(10);

        // 🤖 4️⃣ Création de l’assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();


        //Retrouver les scores des segments/embeddings
        System.out.println("Retrouver les scores des segments/embeddings");
        String question = "Quelle est la signification de RAG ?";  // ou toute autre question
        Embedding embeddingQuestion = embeddingModel.embed(question).content();
        // 2. Construire la requête de recherche des éléments pertinents
        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddingQuestion)
                .maxResults(3)
                .minScore(0.5)
                .build();
        // 3. Récupère les embeddings et segments les plus pertinents :
        EmbeddingSearchResult<TextSegment> embeddingSearchResult = embeddingStore.search(embeddingSearchRequest);
        // 4. Affiche les segments avec leur score :
        System.out.println("Segments avec leur score :");
        for (EmbeddingMatch<TextSegment> match : embeddingSearchResult.matches()) {
            System.out.println("Segment : " + match.embedded() + " avec le score : " + match.score());
        }
        // 🔍 Vérification du RAG avec la même question
        System.out.println("\n=== Vérification : réponse de l'assistant via RAG ===");
        String reponse = assistant.chat(question);
        System.out.println("🤖 Réponse du modèle Gemini (avec RAG) :\n" + reponse);

    } 
}
