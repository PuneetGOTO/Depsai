import discord
import os
import aiohttp
import json
import logging
from collections import deque
import asyncio # 引入 asyncio 用于延迟

# --- 提前设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__) 

# --- 配置 ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner") 
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}") 
TRIGGER_MODE = os.getenv("TRIGGER_MODE", "mention").lower() 
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!ask ") 
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) 
# 添加一个小的延迟配置（秒），用于分割消息发送之间，防止速率限制
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3")) 

# --- DeepSeek API 请求函数 (修改版) ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API，处理 reasoning_content 和 content"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model, 
        "messages": messages,
        # 注意：文档提到 reasoner 模型不支持 temperature, top_p 等参数
    }

    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response_data = await response.json()
            # logger.debug(f"Received response: {json.dumps(response_data, indent=2)}")

            if response.status == 200:
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    # --- 修改点：同时获取 reasoning_content 和 content ---
                    reasoning_content = message_data.get("reasoning_content") 
                    final_content = message_data.get("content")           
                    usage = response_data.get("usage") 

                    # --- 组合成一个字符串用于 Discord 输出 ---
                    full_response_for_discord = ""
                    if reasoning_content:
                        # 可以添加标记，让用户知道这是思维链
                        full_response_for_discord += f"🤔 **思考过程:**\n```\n{reasoning_content.strip()}\n```\n\n" 
                    if final_content:
                        full_response_for_discord += f"💬 **最终回答:**\n{final_content.strip()}"
                    
                    # 如果只有 reasoning_content (虽然不太可能，但做个保护)
                    if not final_content and reasoning_content:
                        full_response_for_discord = reasoning_content.strip() # 直接用思维链

                    if not full_response_for_discord:
                         logger.error("DeepSeek API response missing both reasoning_content and content.")
                         # 返回 None 和错误信息，让调用处知道需要特殊处理
                         return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"

                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    # 返回两个值：用于 Discord 显示的完整组合字符串，和仅用于历史记录的最终答案
                    return full_response_for_discord.strip(), final_content.strip() if final_content else None

                else: # choices 为空或不存在
                    logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                    return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            
            elif response.status == 400: # 特别处理 400 错误，可能因为输入了 reasoning_content
                 error_message = response_data.get("error", {}).get("message", "未知 400 错误")
                 logger.error(f"DeepSeek API error (Status 400 - Bad Request): {error_message}")
                 logger.warning("A 400 error might be caused by including 'reasoning_content' in the input messages. Check conversation history logic.")
                 try:
                    raw_text = await response.text()
                    logger.error(f"Raw error response body (400): {raw_text[:500]}")
                 except Exception: pass
                 return None, f"请求错误 (400): {error_message}\n(请检查是否意外将思考过程加入到了请求历史中)"

            else: # 其他 HTTP 错误
                error_message = response_data.get("error", {}).get("message", "未知错误")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}")
                try:
                    raw_text = await response.text()
                    logger.error(f"Raw error response body ({response.status}): {raw_text[:500]}")
                except Exception: pass
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return None, f"抱歉，无法连接到 DeepSeek API：{e}"
    except json.JSONDecodeError:
        # 尝试读取原始文本来诊断问题
        raw_text_content = "Could not read raw text."
        try:
            # 注意：此时 response 对象可能不在作用域内，取决于异常发生的位置
            # 为了健壮性，这里假设无法直接访问 response.text()
             logger.error("Failed to decode JSON response from DeepSeek API.")
        except Exception as read_err:
             logger.error(f"Failed to decode JSON response and also failed to read raw text: {read_err}")
        return None, f"抱歉，无法解析 DeepSeek API 的响应。"
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return None, f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"


# --- Discord 机器人 ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True 
intents.guilds = True
client = discord.Client(intents=intents)
conversation_history = {}

@client.event
async def on_ready():
    logger.info(f'机器人已登录为 {client.user}')
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}') 
    logger.info(f'Trigger Mode: {TRIGGER_MODE}')
    if TRIGGER_MODE == 'prefix':
        logger.info(f'Command Prefix: "{COMMAND_PREFIX}"')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    print("Bot is ready!") 

@client.event
async def on_message(message):
    """收到消息时执行"""
    if message.author == client.user: return

    triggered = False
    user_prompt = ""
    # (触发逻辑保持不变) ...
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
                 
    if not triggered: return

    channel_id = message.channel.id
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)

    history_deque = conversation_history[channel_id]
    # --- 关键：构建发送给 API 的消息列表，不包含 reasoning_content ---
    api_messages = [] 
    for msg in history_deque:
        # 确保只添加 role 和 content 字段到 API 请求中
        api_messages.append({"role": msg.get("role"), "content": msg.get("content")})
        
    api_messages.append({"role": "user", "content": user_prompt})

    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession() as session:
              logger.info(f"User prompt in channel {channel_id}: {user_prompt}")
              # --- 修改点：接收两个返回值 ---
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      # --- 修改点：处理返回结果 ---
      if response_for_discord: # 如果 API 调用成功并返回了内容
          # --- 关键：只将最终答案 (content) 添加到历史记录 ---
          if response_for_history: # 确保有最终答案才添加到历史
                history_deque.append({"role": "user", "content": user_prompt})
                # 保存 role 和 content 即可，不需要保存 reasoning_content 到历史
                history_deque.append({"role": "assistant", "content": response_for_history}) 
          else:
                # 如果只有思维链没有最终答案，可能不希望将其加入历史，或者只加入用户提问？
                # 这里选择不将只有思维链的回复加入历史，避免污染后续请求
                logger.warning("Response contained reasoning but no final answer. Not adding assistant turn to history.")
                # 可以考虑只把用户消息加入历史？
                # history_deque.append({"role": "user", "content": user_prompt})
          
          # --- 使用现有的分割逻辑发送组合后的完整响应 ---
          if len(response_for_discord) <= 2000:
              await message.channel.send(response_for_discord, reference=message, mention_author=True)
          else:
              logger.warning(f"Combined response is too long ({len(response_for_discord)} chars), splitting.")
              parts = []
              current_pos = 0
              while current_pos < len(response_for_discord):
                  cut_off = min(current_pos + 1990, len(response_for_discord))
                  # 优先找换行符
                  split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                  # 如果找不到换行符，或者最后一部分，或者换行符离当前位置太近，则尝试找空格
                  if split_index == -1 or split_index <= current_pos or cut_off == len(response_for_discord):
                       space_split_index = response_for_discord.rfind(' ', current_pos, cut_off)
                       if space_split_index != -1 and space_split_index > current_pos:
                           split_index = space_split_index
                       else: # 连空格都找不到，或者在最后，硬切
                           split_index = cut_off
                  
                  # 处理代码块（基础版）
                  chunk_to_check = response_for_discord[current_pos:split_index]
                  if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                      fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1) 
                      if fallback_split != -1 and fallback_split > current_pos:
                           split_index = fallback_split

                  parts.append(response_for_discord[current_pos:split_index])
                  current_pos = split_index
                  # 跳过分割处的换行符或空格
                  if current_pos < len(response_for_discord) and response_for_discord[current_pos] in ('\n', ' '):
                      current_pos += 1
              
              # 逐条发送分割后的消息
              for i, part in enumerate(parts):
                  if not part.strip(): 
                      continue
                  ref = message if i == 0 else None
                  mention = True if i == 0 else False
                  # 添加延迟
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0:
                       await asyncio.sleep(SPLIT_MESSAGE_DELAY) 
                  await message.channel.send(part.strip(), reference=ref, mention_author=mention)

      elif response_for_history: # 如果 get_deepseek_response 返回的是错误信息 (第二个返回值)
            await message.channel.send(response_for_history, reference=message, mention_author=True)
      else: 
          # 理论上不应到达这里，因为 get_deepseek_response 会返回错误信息
          logger.error("Received unexpected None values from get_deepseek_response.")
          await message.channel.send("抱歉，处理 DeepSeek 请求时发生未知问题。", reference=message, mention_author=True)
          
    except Exception as e:
        logger.exception(f"An error occurred in on_message handler for message ID {message.id}")
        await message.channel.send(f"处理你的请求时发生内部错误：{e}", reference=message, mention_author=True)

# --- 运行 Bot ---
if __name__ == "__main__":
    # (启动检查保持不变) ...
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