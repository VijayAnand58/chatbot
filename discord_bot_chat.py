from discord.ext import commands

import discord

from chatbot import ask_question
import os
from dotenv import load_dotenv
load_dotenv()

CHANNEL_ID=int(os.getenv("CHANNEL_ID"))
TOKEN=os.getenv("DISCORD_API")

bot=commands.Bot(command_prefix='!',intents=discord.Intents.all())

@bot.event
async def on_ready():
    print("Bot is initialized")
    channel=bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("""
                           Hi Welcome to Organic Chemistry Bot,
                           \n Here you can ask questions to the bot regading organic chemistry and it will answer.
                           \n It reffers to the works of  Dr. C. Neuman, Jr., Professor of Chemistry, University of California, Riverside.
                           \n The bot's context is confined to the knowledge in the textbook.""")
    else:
        print("Channel not found!")

@bot.command()
async def q(ctx,*,query : str = None):
    if query:
        answer=ask_question(query)
        if answer:
            await ctx.send(answer)
        else:
            await ctx.send("Try again after 5 seconds")

bot.run(TOKEN)

