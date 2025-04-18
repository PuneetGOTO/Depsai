import discord
import os
import aiohttp
import json
import logging
from collections import deque

# --- 提前设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__) # 获取logger实例

# --- 配置 ---
# 从环境变量获取 Token 和 Key
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 使用 'deepseek-reasoner' 作为默认模型
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner") 
# 现在可以安全地记录日志了
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}") 

# 触发机器人的方式：'mention' (提及) 或 'prefix' (前缀)
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "mention").lower() 
# 如果 TRIGGER_MODE 是 'prefix', 使用这个前缀
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!ask ") 
# 允许的最大对话历史轮数 (用户 + 机器人)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) 

# --- DeepSeek API 请求函数 ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model, 
        "messages": messages,
    }

    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response_data = await response.json()
            # logger.debug(f"Received response: {json.dumps(response_data, indent=2)}")

            if response.status == 200:
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0].get("message", {}).get("content")
                    usage = response_data.get("usage") 
                    if content:
                        logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                        return content.strip()
                    else:
                        logger.error("DeepSeek API response missing content.")
                        return "抱歉，DeepSeek API 返回的数据似乎不完整。"
                else:
                    logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                    return f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else:
                error_message = response_data.get("error", {}).get("message", "未知错误")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}")
                try:
                    raw_text = await response.text()
                    logger.error(f"Raw error response body: {raw_text[:500]}")
                except Exception:
                    pass 
                return f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return f"抱歉，无法连接到 DeepSeek API：{e}"
    except json.JSONDecodeError:
        try:
            # 尝试在 json 解码失败时读取原始文本
            raw_text = await response.text() # 注意：response 可能在这里未定义，需要调整逻辑或确保能访问
            logger.error(f"Failed to decode JSON response from DeepSeek API. Raw response text: {raw_text[:500]}...")
        except Exception:
             logger.error("Failed to decode JSON response from DeepSeek API and couldn't read raw text.")
        return "抱歉，无法解析 DeepSeek API 的响应。"
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"

# --- Discord 机器人 ---

# 设置 Intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True 
intents.guilds = True

client = discord.Client(intents=intents)

# 对话历史
conversation_history = {}

@client.event
async def on_ready():
    """机器人启动时执行"""
    logger.info(f'机器人已登录为 {client.user}')
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}') 
    logger.info(f'Trigger Mode: {TRIGGER_MODE}')
    if TRIGGER_MODE == 'prefix':
        logger.info(f'Command Prefix: "{COMMAND_PREFIX}"')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    print("Bot is ready!") 

@client.event
async def on_message(message):
    """收到消息时执行"""
    if message.author == client.user:
        return

    triggered = False
    user_prompt = ""

    if TRIGGER_MODE == 'mention':
        if client.user.mentioned_in(message):
            triggered = True
            user_prompt = message.content.replace(f'<@!{client.user.id}>', '', 1).replace(f'<@{client.user.id}>', '', 1).strip()
            if not user_prompt:
                 await message.channel.send("你好！有什么可以帮你的吗？请 @我 并输入你的问题。", reference=message, mention_author=True)
                 return
    elif TRIGGER_MODE == 'prefix':
        if message.content.startswith(COMMAND_PREFIX):
            triggered = True
            user_prompt = message.content[len(COMMAND_PREFIX):].strip()
            if not user_prompt:
                 await message.channel.send(f"请输入内容。用法：`{COMMAND_PREFIX}你的问题`", reference=message, mention_author=True)
                 return
                 
    if not triggered:
        return

    channel_id = message.channel.id
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)

    history_deque = conversation_history[channel_id]
    api_messages = list(history_deque) 
    api_messages.append({"role": "user", "content": user_prompt})

    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession() as session:
              logger.info(f"User prompt in channel {channel_id}: {user_prompt}")
              response_content = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      if response_content and "抱歉" not in response_content:
          history_deque.append({"role": "user", "content": user_prompt})
          history_deque.append({"role": "assistant", "content": response_content}) 
          
          if len(response_content) <= 2000:
              await message.channel.send(response_content, reference=message, mention_author=True)
          else:
              logger.warning(f"Response is too long ({len(response_content)} chars), splitting.")
              parts = []
              current_pos = 0
              while current_pos < len(response_content):
                  cut_off = min(current_pos + 1990, len(response_content))
                  split_index = response_content.rfind('\n', current_pos, cut_off)

                  if split_index == -1 or split_index <= current_pos or len(response_content) - current_pos <= 1990:
                    split_index = cut_off
                  
                  # 简化版代码块检查
                  chunk_to_check = response_content[current_pos:split_index]
                  if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                      fallback_split = response_content.rfind('\n', current_pos, split_index - 1) 
                      if fallback_split != -1 and fallback_split > current_pos:
                           split_index = fallback_split
                      
                  parts.append(response_content[current_pos:split_index])
                  current_pos = split_index
                  if current_pos < len(response_content) and response_content[current_pos] == '\n':
                      current_pos += 1

              for i, part in enumerate(parts):
                  if not part.strip(): 
                      continue
                  ref = message if i == 0 else None
                  mention = True if i == 0 else False
                  await message.channel.send(part.strip(), reference=ref, mention_author=mention)

      elif response_content: 
            await message.channel.send(response_content, reference=message, mention_author=True)
      else:
          logger.error("Received empty or None response from get_deepseek_response unexpectedly.")
          await message.channel.send("抱歉，从 DeepSeek 获取回复时发生未知问题。", reference=message, mention_author=True)
          
    except Exception as e:
        logger.exception(f"An error occurred in on_message handler for message ID {message.id}")
        await message.channel.send(f"处理你的请求时发生内部错误：{e}", reference=message, mention_author=True)


# --- 运行 Bot ---
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        logger.critical("错误：未找到 Discord Bot Token (环境变量 DISCORD_BOT_TOKEN)")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if not DEEPSEEK_API_KEY:
        logger.critical("错误：未找到 DeepSeek API Key (环境变量 DEEPSEEK_API_KEY)")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人...")
        logger.info(f"Attempting to run with model: {DEEPSEEK_MODEL}") 
        client.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("Discord Bot Token 无效，登录失败。请检查你的 Token。")
    except discord.PrivilegedIntentsRequired:
        logger.critical("Message Content Intent 未启用！请前往 Discord Developer Portal -> Bot -> Privileged Gateway Intents 开启 Message Content Intent。")
    except Exception as e:
        logger.exception("启动机器人时发生未捕获的错误。")