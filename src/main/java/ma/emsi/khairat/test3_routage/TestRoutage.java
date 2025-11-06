package ma.emsi.khairat.test3_routage;

import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import ma.emsi.khairat.test1_ragnaif.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.Handler;

public class TestRoutage {

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
        configureLogger();
        System.out.println("=== Test 3 : Routage ===");

        // 1️⃣ Parser + modèle d’embedding
        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2️⃣ Charger et découper les 2 documents
        List<TextSegment> segmentsIA = loadAndSplit("src/main/resources/rag-2.pdf", parser);
        List<TextSegment> segmentsSport = loadAndSplit("src/main/resources/sport.pdf", parser);

        // 3️⃣ Créer 2 magasins + y ajouter les embeddings
        EmbeddingStore<TextSegment> storeIA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeSport = new InMemoryEmbeddingStore<>();

        storeIA.addAll(embeddingModel.embedAll(segmentsIA).content(), segmentsIA);
        storeSport.addAll(embeddingModel.embedAll(segmentsSport).content(), segmentsSport);

        // 4️⃣ Créer les 2 retrievers
        var retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        var retrieverSport = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeSport)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // 5️⃣ Modèle Gemini
        String key = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 6️⃣ Routage : description de chaque source
        Map<ContentRetriever, String> desc = new HashMap<>();
        desc.put(retrieverIA, "Documents de cours sur le RAG, le fine-tuning et l'intelligence artificielle");
        desc.put(retrieverSport, "Articles sur le sport, la santé et l'entraînement physique");


        var queryRouter = new LanguageModelQueryRouter(model, desc);

        // 7️⃣ Créer le RetrievalAugmentor basé sur le routeur
        var augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 8️⃣ Créer l’assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        // 9️⃣ Tester avec des questions différentes
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("\n👤 Vous : ");
            String question = sc.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            String reponse = assistant.chat(question);
            System.out.println("🤖 Gemini : " + reponse);
        }
    }

    private static List<TextSegment> loadAndSplit(String chemin, DocumentParser parser) {
        Path path = Paths.get(chemin);
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
        return DocumentSplitters.recursive(300, 30).split(doc);
    }
}
