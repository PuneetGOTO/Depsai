import discord
from discord.ext import commands
from discord import app_commands # 用于斜杠命令
from discord.ui import View, Button, button # 用于按钮和视图
import os
import aiohttp
import json
import logging
from collections import deque
import asyncio
import re # 用于清理用户名

# --- 提前设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3"))
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "")
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
PRIVATE_CHANNEL_PREFIX = "deepseek-"

# --- DeepSeek API 请求函数 (保持不变) ---
async def get_deepseek_response(session, api_key, model, messages):
    # ... (此处省略，使用上一个版本完整的 get_deepseek_response 代码) ...
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": messages}
    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300) as response: # 增加超时
            raw_response_text = await response.text()
            try:
                response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response from DeepSeek API. Status: {response.status}. Raw text: {raw_response_text[:500]}...")
                 return None, f"抱歉，无法解析 DeepSeek API 的响应 (状态码 {response.status})。"

            if response.status == 200:
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    reasoning_content = message_data.get("reasoning_content")
                    final_content = message_data.get("content")
                    usage = response_data.get("usage")
                    full_response_for_discord = ""
                    if reasoning_content:
                        full_response_for_discord += f"🤔 **思考过程:**\n```\n{reasoning_content.strip()}\n```\n\n"
                    if final_content:
                        full_response_for_discord += f"💬 **最终回答:**\n{final_content.strip()}"
                    if not final_content and reasoning_content:
                         full_response_for_discord = reasoning_content.strip()
                    if not full_response_for_discord:
                         logger.error("DeepSeek API response missing both reasoning_content and content.")
                         return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"
                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    return full_response_for_discord.strip(), final_content.strip() if final_content else None
                 else:
                     logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                     return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else: # 处理错误状态码
                error_message = response_data.get("error", {}).get("message", f"未知错误 (状态码 {response.status})")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return None, f"抱歉，无法连接到 DeepSeek API：{e}"
    except asyncio.TimeoutError:
        logger.error("Request to DeepSeek API timed out.")
        return None, "抱歉，连接 DeepSeek API 超时。"
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return None, f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"


# --- Discord 机器人设置 ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 对话历史
conversation_history = {}

# --- 创建按钮视图 ---
class CreateChatView(View):
    # timeout=None 使视图持久化，按钮在机器人重启后依然有效
    # 但这需要更复杂的处理（通常结合数据库存储message_id并在启动时重新添加视图）
    # 为了简单，我们先不设置 timeout=None，按钮将在机器人重启后失效
    # 如果需要持久化，请取消注释下面这行并实现 add_view 的逻辑
    # def __init__(self):
    #    super().__init__(timeout=None)

    @button(label="创建私密聊天", style=discord.ButtonStyle.primary, emoji="💬", custom_id="create_deepseek_chat_button")
    async def create_chat_button_callback(self, interaction: discord.Interaction, button_obj: Button):
        """按钮被点击时执行的回调函数"""
        guild = interaction.guild
        user = interaction.user
        bot_member = guild.get_member(bot.user.id) # 在回调中重新获取 bot member

        if not guild:
            await interaction.response.send_message("此操作只能在服务器内进行。", ephemeral=True)
            return
            
        if not bot_member or not bot_member.guild_permissions.manage_channels:
            await interaction.response.send_message("机器人缺少“管理频道”权限，无法创建聊天频道。", ephemeral=True)
            logger.warning(f"Button Click: Missing 'Manage Channels' permission in guild {guild.id}")
            return

        # --- 以下逻辑与之前的 /new_chat 基本相同 ---
        clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower()
        if not clean_user_name: clean_user_name = "user"
        channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}"

        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False),
            user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True),
            bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True)
        }
        for role_id in ADMIN_ROLE_IDS:
            role = guild.get_role(role_id)
            if role:
                overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True)

        try:
            # 告知用户正在创建 (ephemeral 对按钮点击者可见)
            await interaction.response.send_message(f"正在为你创建私密聊天频道 **{channel_name}** ...", ephemeral=True)

            category = discord.utils.find(lambda c: c.name.lower() == "deepseek chats", guild.categories)
            if category and category.permissions_for(bot_member).manage_channels:
                 new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=category)
            else:
                 new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites)

            logger.info(f"Button Click: Created private channel {new_channel.name} ({new_channel.id}) for user {user.name} ({user.id})")

            await new_channel.send(
                f"你好 {user.mention}！\n"
                f"欢迎来到你的 DeepSeek 私密聊天频道 (使用模型: **{DEEPSEEK_MODEL}**)。\n"
                f"直接在此输入你的问题即可开始对话。\n"
                f"对话历史最多保留 **{MAX_HISTORY // 2}** 轮问答。\n"
                f"当你完成后，可以在此频道使用 `/close_chat` 命令来关闭它。"
            )

            # 用 followup 发送频道链接，因为上面已经 response
            await interaction.followup.send(f"你的私密聊天频道已创建：{new_channel.mention}", ephemeral=True)

        except discord.Forbidden:
            logger.error(f"Button Click: Permission error (Forbidden) for user {user.id} creating channel '{channel_name}'.")
            try: await interaction.followup.send("创建频道失败：权限不足。", ephemeral=True)
            except discord.NotFound: pass
        except discord.HTTPException as e:
            logger.error(f"Button Click: HTTP error for user {user.id} creating channel '{channel_name}': {e}")
            try: await interaction.followup.send(f"创建频道时发生网络错误。", ephemeral=True)
            except discord.NotFound: pass
        except Exception as e:
            logger.exception(f"Button Click: Unexpected error for user {user.id} creating channel")
            try: await interaction.followup.send(f"创建频道时发生未知错误。", ephemeral=True)
            except discord.NotFound: pass


# --- setup_hook 同步斜杠命令和打印配置 ---
# 注意：如果需要持久化视图，需要在这里添加 bot.add_view() 的逻辑
@bot.event
async def setup_hook():
    logger.info("Running setup_hook...")
    # --- 打印配置日志 ---
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Admin Role IDs: {ADMIN_ROLE_IDS}')
    logger.info(f'Private Channel Prefix: {PRIVATE_CHANNEL_PREFIX}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    
    # --- 注册视图 (如果需要持久化) ---
    # view = CreateChatView() # 如果 CreateChatView 的 __init__ 设置了 timeout=None
    # bot.add_view(view) # 这通常需要配合 message_id，逻辑较复杂，暂不实现
    # logger.info("Persistent views (if any) registered.")
    
    # --- 同步斜杠命令 ---
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands globally: {[c.name for c in synced]}")
    except Exception as e:
        logger.exception(f"Failed to sync slash commands: {e}")

@bot.event
async def on_ready():
    logger.info(f'机器人已登录为 {bot.user}')
    print(f"Logged in as {bot.user} ({bot.user.id})")
    print("Bot is ready.")


# --- 新增：发送带按钮面板的命令 ---
@bot.tree.command(name="setup_panel", description="发送一个包含'创建聊天'按钮的消息到当前频道")
@app_commands.checks.has_permissions(manage_guild=True) # 限制管理员使用
async def setup_panel(interaction: discord.Interaction, message_content: str = "点击下面的按钮开始与 DeepSeek 的私密聊天："):
    """发送包含创建聊天按钮的消息"""
    try:
        view = CreateChatView() # 创建一个新的视图实例
        await interaction.channel.send(message_content, view=view)
        await interaction.response.send_message("创建聊天按钮面板已发送！", ephemeral=True)
        logger.info(f"User {interaction.user} deployed the create chat panel in channel {interaction.channel.id}")
    except discord.Forbidden:
         logger.error(f"Failed to send setup panel in {interaction.channel.id}: Missing permissions.")
         await interaction.response.send_message("发送失败：机器人缺少在此频道发送消息或添加组件的权限。", ephemeral=True)
    except Exception as e:
        logger.exception(f"Failed to send setup panel in {interaction.channel.id}")
        await interaction.response.send_message(f"发送面板时发生错误：{e}", ephemeral=True)

@setup_panel.error # 错误处理 for setup_panel
async def setup_panel_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
     if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("你需要“管理服务器”权限才能使用此命令。", ephemeral=True)
     else:
        logger.error(f"Error in setup_panel command: {error}")
        await interaction.response.send_message("执行命令时发生未知错误。", ephemeral=True)


# --- 斜杠命令：关闭聊天频道 (保持不变) ---
@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
@app_commands.checks.has_permissions(send_messages=True)
async def close_chat(interaction: discord.Interaction):
    # ... (此处省略，使用上一个版本完整的 close_chat 代码) ...
    channel = interaction.channel
    user = interaction.user
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return
    try:
        await interaction.response.send_message(f"请求收到！频道 {channel.mention} 将在几秒后关闭...", ephemeral=True)
        await channel.send(f"此聊天频道由 {user.mention} 请求关闭，将在 5 秒后删除。")
        logger.info(f"User {user.name} ({user.id}) initiated closure of channel {channel.name} ({channel.id})")
        if channel.id in conversation_history:
            del conversation_history[channel.id]
            logger.info(f"Removed conversation history for channel {channel.id}")
        await asyncio.sleep(5)
        await channel.delete(reason=f"Closed by user {user.name} ({user.id}) via /close_chat")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")
    except discord.Forbidden:
        logger.error(f"Permission error (Forbidden) while trying to delete channel {channel.id}.")
        try: await interaction.followup.send("关闭频道失败：机器人缺少“管理频道”权限。", ephemeral=True)
        except discord.NotFound: pass
    except discord.NotFound:
        logger.warning(f"Channel {channel.id} not found during deletion.")
    except Exception as e:
        logger.exception(f"Unexpected error during /close_chat command for channel {channel.id}")

# --- on_message 事件处理 (保持不变) ---
@bot.event
async def on_message(message: discord.Message):
    # ... (此处省略，使用上一个版本完整的 on_message 代码) ...
    if message.author == bot.user or message.author.bot: return
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX): return

    user_prompt = message.content.strip()
    if not user_prompt: return

    channel_id = message.channel.id
    logger.info(f"Handling message in private channel {message.channel.name} ({channel_id}) from {message.author}: '{user_prompt[:50]}...'")

    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)

    history_deque = conversation_history[channel_id]
    api_messages = []
    for msg in history_deque:
        api_messages.append({"role": msg.get("role"), "content": msg.get("content")})
    api_messages.append({"role": "user", "content": user_prompt})

    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      if response_for_discord:
          if response_for_history:
                history_deque.append({"role": "user", "content": user_prompt})
                history_deque.append({"role": "assistant", "content": response_for_history})
          else:
                logger.warning(f"Response for channel {channel_id} lacked final content.")

          if len(response_for_discord) <= 2000:
              await message.channel.send(response_for_discord)
          else: # 分割逻辑
              logger.warning(f"Response for channel {channel_id} is too long ({len(response_for_discord)} chars), splitting.")
              parts = []
              current_pos = 0
              # ... (省略详细分割代码, 与上一版本相同) ...
              while current_pos < len(response_for_discord):
                    cut_off = min(current_pos + 1990, len(response_for_discord))
                    split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                    if split_index == -1 or split_index <= current_pos:
                        space_split_index = response_for_discord.rfind(' ', current_pos, cut_off)
                        if space_split_index != -1 and space_split_index > current_pos: split_index = space_split_index
                        else: split_index = cut_off
                    chunk_to_check = response_for_discord[current_pos:split_index]
                    if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                        fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1)
                        if fallback_split != -1 and fallback_split > current_pos: split_index = fallback_split
                    parts.append(response_for_discord[current_pos:split_index])
                    current_pos = split_index
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace(): current_pos += 1
              for i, part in enumerate(parts):
                  if not part.strip(): continue
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0: await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  await message.channel.send(part.strip())

      elif response_for_history: # API 返回错误信息
            await message.channel.send(f"抱歉，处理你的请求时发生错误：\n{response_for_history}")
      else: # 未知错误
          logger.error(f"Received unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时发生未知问题。")
    except Exception as e:
        logger.exception(f"An error occurred in on_message handler for channel {channel_id}")
        try: await message.channel.send(f"处理你的消息时发生内部错误: {e}")
        except Exception: pass

# --- 运行 Bot ---
if __name__ == "__main__":
    # ... (启动检查保持不变) ...
    if not DISCORD_BOT_TOKEN:
        logger.critical("错误：未找到 Discord Bot Token (环境变量 DISCORD_BOT_TOKEN)")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if not DEEPSEEK_API_KEY:
        logger.critical("错误：未找到 DeepSeek API Key (环境变量 DEEPSEEK_API_KEY)")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人 (commands.Bot)...")
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure:
        logger.critical("Discord Bot Token 无效，登录失败。请检查你的 Token。")
    except discord.PrivilegedIntentsRequired as e:
        logger.critical(f"必需的 Intents 未启用！请开启 Message Content Intent。错误: {e}")
    except Exception as e:
        logger.exception("启动机器人时发生未捕获的错误。")