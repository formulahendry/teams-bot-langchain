import { Activity, CardFactory, MessageFactory, TurnContext } from "botbuilder";
import { CommandMessage, TeamsFxBotCommandHandler, TriggerPatterns } from "@microsoft/teamsfx";
import { AdaptiveCards } from "@microsoft/adaptivecards-tools";
import helloWorldCard from "./adaptiveCards/helloworldCommand.json";
import { CardData } from "./cardModels";
import { langchain } from "./ai/langchain";

/**
 * The `HelloWorldCommandHandler` registers a pattern with the `TeamsFxBotCommandHandler` and responds
 * with an Adaptive Card if the user types the `triggerPatterns`.
 */
export class HelloWorldCommandHandler implements TeamsFxBotCommandHandler {
  triggerPatterns: TriggerPatterns = new RegExp(".+");

  async handleCommandReceived(
    context: TurnContext,
    message: CommandMessage
  ): Promise<string | Partial<Activity> | void> {
    console.log(`App received message: ${message.text}`);

    // C:/Users/junhan.FAREAST/Downloads/semantic-kernel-1-6.pdf
    if (message.text.includes(".pdf")) {
      context.sendActivity("File uploading...");
      await langchain.uploadFile(message.text);
      return MessageFactory.text("File uploaded.");
    } else {
      const answer = await langchain.askQuestion(message.text);
      return MessageFactory.text(answer);
    }
  }
}
