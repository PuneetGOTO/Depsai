# -*- coding: utf-8 -*-
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
# 配置日志记录器，设定级别为 INFO，并指定格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
# 获取当前模块的 logger 实例
logger = logging.getLogger(__name__)

# --- 配置 ---
# 从环境变量获取敏感信息和配置
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek API 端点

# --- 可用模型定义 ---
# 包含模型ID、描述和是否支持视觉输入
# !!重要!!: 请根据 DeepSeek 最新官方文档确认模型 ID 和 supports_vision 的准确性
AVAILABLE_MODELS = {
    "deepseek-chat": {
        "description": "通用对话模型，平衡性能和速度，支持文本和图像(Vision)。",
        "supports_vision": True, # 假设 deepseek-chat 支持视觉
    },
    "deepseek-coder": {
        "description": "代码生成和理解模型，专注于编程任务。",
        "supports_vision": False, # 通常代码模型不支持视觉
    },
    "deepseek-reasoner": {
        "description": "推理模型，擅长复杂逻辑、数学和思维链输出。",
        "supports_vision": False, # 假设推理模型不支持直接视觉输入
    },
    # 可以根据需要添加更多模型
    # "some-vision-model-id": {
    #     "description": "专门的视觉模型",
    #     "supports_vision": True,
    # }
}
# --- 设置默认和当前模型 ---
DEFAULT_MODEL_ID = "deepseek-chat" # 将支持视觉的设为默认，或者第一个可用模型
# 检查环境变量中的模型设置
initial_model_id = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL_ID)
if initial_model_id not in AVAILABLE_MODELS:
    logger.warning(f"环境指定的模型 '{initial_model_id}' 不在可用列表中，将使用默认模型 '{DEFAULT_MODEL_ID}'。")
    initial_model_id = DEFAULT_MODEL_ID
# 全局变量存储当前激活的模型ID
current_model_id = initial_model_id
logger.info(f"Initializing with DeepSeek Model: {current_model_id}")

# --- 其他配置 ---
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) # 总历史轮数
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3")) # 分割消息延迟
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "") # 管理员角色ID列表
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
PRIVATE_CHANNEL_PREFIX = "deepseek-" # 私密频道前缀
MAX_IMAGE_ATTACHMENTS = int(os.getenv("MAX_IMAGE_ATTACHMENTS", 3)) # 最大图片附件数

# --- DeepSeek API 请求函数 ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API，可以处理包含图像 URL 的消息"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        # 可以根据模型调整 max_tokens, 但 Vision 模型可能需要更大值
        # "max_tokens": 4096
    }
    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}") # 调试时取消注释

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300) as response:
            raw_response_text = await response.text()
            try: response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed JSON decode. Status: {response.status}. Text: {raw_response_text[:500]}...");
                 return None, f"无法解析响应(状态{response.status})"

            if response.status == 200:
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    final_content = message_data.get("content") # 主要获取 content
                    usage = response_data.get("usage")
                    if not final_content: logger.error("API response missing content."); return None, "响应数据不完整。"
                    logger.info(f"Success. Usage: {usage}")
                    # 返回显示内容和历史内容（对于非reasoner模型通常相同）
                    return final_content.strip(), final_content.strip()
                 else: logger.error(f"API response missing 'choices': {response_data}"); return None, f"意外结构：{response_data}"
            else:
                error_message = response_data.get("error", {}).get("message", f"未知错误(状态{response.status})")
                logger.error(f"API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                if response.status == 400: error_message += "\n(提示: 400 通常因格式错误或输入无效)"
                return None, f"API 调用出错 (状态{response.status}): {error_message}"
    except aiohttp.ClientConnectorError as e: logger.error(f"Network error: {e}"); return None, "无法连接 API"
    except asyncio.TimeoutError: logger.error("API request timed out."); return None, "API 连接超时"
    except Exception as e: logger.exception("Unexpected API call error."); return None, f"未知 API 错误: {e}"

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
    # 按钮是非持久化的
    @button(label="创建私密聊天", style=discord.ButtonStyle.primary, emoji="💬", custom_id="create_deepseek_chat_button")
    async def create_chat_button_callback(self, interaction: discord.Interaction, button_obj: Button):
        guild = interaction.guild; user = interaction.user; bot_member = guild.get_member(bot.user.id) if guild else None
        if not guild or not bot_member: await interaction.response.send_message("无法获取服务器信息。", ephemeral=True); return
        if not bot_member.guild_permissions.manage_channels: await interaction.response.send_message("机器人缺少“管理频道”权限。", ephemeral=True); return

        clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower() or "user"
        channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}"

        overwrites = { guild.default_role: discord.PermissionOverwrite(read_messages=False), user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True), bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True) }
        for role_id in ADMIN_ROLE_IDS: role = guild.get_role(role_id); overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True) if role else None

        try:
            await interaction.response.send_message(f"正在创建频道 **{channel_name}** ...", ephemeral=True)
            category_name = "DeepSeek Chats"; category = discord.utils.find(lambda c: c.name.lower() == category_name.lower(), guild.categories)
            target_category = category if category and isinstance(category, discord.CategoryChannel) and category.permissions_for(bot_member).manage_channels else None
            new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=target_category)
            logger.info(f"Button Click: Created channel {new_channel.name} for user {user.name}")
            # 欢迎消息显示当前模型
            await new_channel.send(f"你好 {user.mention}！\n欢迎来到 DeepSeek 私密聊天频道 (当前模型: **{current_model_id}**)。\n直接输入问题或上传图片并提问（如果模型支持）。\n历史最多保留 **{MAX_HISTORY // 2}** 轮。\n完成后可用 `/close_chat` 关闭。")
            await interaction.followup.send(f"频道已创建：{new_channel.mention}", ephemeral=True)
        except Exception as e: logger.exception(f"Button Click: Error creating channel for {user.id}"); await interaction.followup.send("创建频道时出错。", ephemeral=True)

# --- setup_hook ---
@bot.event
async def setup_hook():
    logger.info("Running setup_hook...")
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Current Active DeepSeek Model: {current_model_id}') # 显示当前模型
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Admin Role IDs: {ADMIN_ROLE_IDS}')
    logger.info(f'Private Channel Prefix: {PRIVATE_CHANNEL_PREFIX}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands globally: {[c.name for c in synced]}") # 列出同步的命令
    except Exception as e: logger.exception("Failed to sync slash commands")

# --- on_ready ---
@bot.event
async def on_ready():
     logger.info(f'机器人已登录为 {bot.user}')
     print("Bot is ready and functional.")

# --- 斜杠命令 ---

# /setup_panel
@bot.tree.command(name="setup_panel", description="发送一个包含'创建聊天'按钮的消息到当前频道")
async def setup_panel(interaction: discord.Interaction, message_content: str = "点击下面的按钮开始与 DeepSeek 的私密聊天："):
    channel = interaction.channel; user = interaction.user
    if not interaction.guild: await interaction.response.send_message("此命令只能在服务器频道中使用。", ephemeral=True); return
    is_private = isinstance(channel, discord.TextChannel) and channel.name.startswith(PRIVATE_CHANNEL_PREFIX)
    can_execute = is_private or (isinstance(user, discord.Member) and user.guild_permissions.manage_guild)
    if not can_execute: await interaction.response.send_message("你需要在非私密频道中拥有“管理服务器”权限。", ephemeral=True); return
    try:
        await channel.send(message_content, view=CreateChatView()); await interaction.response.send_message("按钮面板已发送！", ephemeral=True)
    except Exception as e: logger.exception(f"Failed setup panel in {channel.id}"); await interaction.response.send_message(f"发送面板出错: {e}", ephemeral=True)

# /clear_history
@bot.tree.command(name="clear_history", description="清除当前私密聊天频道的对话历史")
async def clear_history(interaction: discord.Interaction):
    channel = interaction.channel; user = interaction.user
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX): await interaction.response.send_message("此命令只能在私密频道中使用。", ephemeral=True); return
    channel_id = channel.id
    if channel_id in conversation_history:
        try: conversation_history[channel_id].clear(); logger.info(f"User {user} cleared history for {channel_id}"); await interaction.response.send_message("对话历史已清除。", ephemeral=False)
        except Exception as e: logger.exception(f"Error clearing history {channel_id}"); await interaction.response.send_message(f"清除历史出错: {e}", ephemeral=True)
    else: await interaction.response.send_message("未找到历史记录。", ephemeral=True)

# /help (显示当前模型)
@bot.tree.command(name="help", description="显示机器人使用帮助")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="DeepSeek 机器人帮助", description=f"**当前激活模型:** `{current_model_id}`", color=discord.Color.purple())
    embed.add_field(name="开始聊天", value="点击 **“创建私密聊天”** 按钮创建专属频道。", inline=False)
    embed.add_field(name="在私密频道中", value=f"• 直接输入问题或上传图片并提问（如果当前模型支持）。\n• 最多保留 **{MAX_HISTORY // 2}** 轮历史。", inline=False)
    embed.add_field(name="可用命令", value="`/help`: 显示此帮助。\n`/list_models`: 查看可用模型。\n`/set_model <model_id>`: (管理员) 切换模型。\n`/clear_history`: (私密频道内) 清除历史。\n`/close_chat`: (私密频道内) 关闭频道。\n`/setup_panel`: 发送创建按钮面板。", inline=False)
    embed.set_footer(text=f"模型: {current_model_id}")
    await interaction.response.send_message(embed=embed, ephemeral=True)

# /list_models
@bot.tree.command(name="list_models", description="查看可用 DeepSeek 模型及当前激活模型")
async def list_models(interaction: discord.Interaction):
    embed = discord.Embed(title="可用 DeepSeek 模型", description=f"**当前激活模型:** `{current_model_id}` ✨", color=discord.Color.green())
    for model_id, info in AVAILABLE_MODELS.items():
        vision_support = "✅ 支持视觉" if info["supports_vision"] else "❌ 不支持视觉"
        embed.add_field(name=f"`{model_id}`", value=f"{info['description']}\n*{vision_support}*", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

# /set_model
@bot.tree.command(name="set_model", description="[管理员] 切换机器人使用的 DeepSeek 模型")
@app_commands.describe(model_id="要切换到的模型 ID")
@app_commands.choices(model_id=[app_commands.Choice(name=mid, value=mid) for mid in AVAILABLE_MODELS.keys()])
@app_commands.checks.has_permissions(manage_guild=True)
async def set_model(interaction: discord.Interaction, model_id: app_commands.Choice[str]):
    global current_model_id
    chosen_model = model_id.value
    if chosen_model == current_model_id: await interaction.response.send_message(f"机器人当前已在使用 `{chosen_model}`。", ephemeral=True); return
    if chosen_model in AVAILABLE_MODELS:
        current_model_id = chosen_model
        logger.info(f"User {interaction.user} changed active model to: {current_model_id}")
        await interaction.response.send_message(f"✅ 模型已切换为: `{current_model_id}`", ephemeral=False)
    else: await interaction.response.send_message(f"❌ 错误：无效模型 ID。", ephemeral=True)

@set_model.error
async def set_model_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
     if isinstance(error, app_commands.MissingPermissions): await interaction.response.send_message("你需要“管理服务器”权限才能切换模型。", ephemeral=True)
     else: logger.error(f"Error in set_model: {error}"); await interaction.response.send_message("执行命令出错。", ephemeral=True)

# /close_chat (使用修正后的代码)
@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
async def close_chat(interaction: discord.Interaction):
    channel = interaction.channel; user = interaction.user
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX): await interaction.response.send_message("此命令只能在私密频道中使用。", ephemeral=True); return
    try:
        await interaction.response.send_message(f"频道将在几秒后关闭...", ephemeral=True)
        await channel.send(f"此频道由 {user.mention} 请求关闭，将在 5 秒后删除。")
        channel_id = channel.id # 获取 ID
        if channel_id in conversation_history:
            try: del conversation_history[channel_id]; logger.info(f"Removed history for {channel_id}")
            except KeyError: logger.warning(f"History key {channel_id} already gone.")
        else: logger.warning(f"No history found for {channel_id} during closure.")
        await asyncio.sleep(5); await channel.delete(reason=f"Closed by {user}")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")
    except Exception as e: logger.exception(f"Error closing channel {channel.id}")


# --- on_message 事件处理 (核心对话逻辑，含视觉判断) ---
@bot.event
async def on_message(message: discord.Message):
    """处理在私密频道中接收到的消息，根据当前模型处理文本和图片"""
    if message.author == bot.user or message.author.bot: return
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX): return
    if message.content.strip().startswith('/'): return # 忽略命令

    channel_id = message.channel.id
    user_prompt_text = message.content.strip()
    image_urls = []
    ignored_images = False

    # --- 根据当前模型决定是否处理图片 ---
    model_info = AVAILABLE_MODELS.get(current_model_id, {"supports_vision": False})
    supports_vision = model_info["supports_vision"]

    if message.attachments:
        processed_images = 0
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                if supports_vision and processed_images < MAX_IMAGE_ATTACHMENTS:
                    image_urls.append(attachment.url); processed_images += 1
                elif supports_vision: ignored_images = True; break
                else: ignored_images = True # 模型不支持，忽略

    # 如果图片被忽略，发送提示
    if ignored_images:
        reason = f"本次只处理了前 {MAX_IMAGE_ATTACHMENTS} 张图片" if supports_vision else f"当前模型 `{current_model_id}` 不支持处理图片"
        try: await message.reply(f"注意：{reason}，已忽略多余或不支持的附件。", mention_author=False, delete_after=20)
        except discord.HTTPException: pass # 忽略发送提示失败的情况

    # 检查有效输入
    if not user_prompt_text and not image_urls: return # 无有效输入则忽略
    if not user_prompt_text and image_urls: user_prompt_text = "描述这张/这些图片。" # 只有图片时的默认提示

    logger.info(f"Handling message in {channel_id} from {message.author} with text: '{user_prompt_text[:50]}...' and {len(image_urls)} image(s). Using model: {current_model_id}")

    # 获取或初始化历史
    if channel_id not in conversation_history: conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)
    history_deque = conversation_history[channel_id]

    # --- 构建 API 消息列表 ---
    api_messages = []
    for msg in history_deque: api_messages.append({"role": msg.get("role"), "content": msg.get("content")}) # 仅文本历史

    # 构建当前用户消息 (多模态或纯文本)
    if supports_vision and image_urls:
        current_user_message_content = [{"type": "text", "text": user_prompt_text}]
        for url in image_urls: current_user_message_content.append({"type": "image_url", "image_url": {"url": url}})
        api_messages.append({"role": "user", "content": current_user_message_content})
    else:
        api_messages.append({"role": "user", "content": user_prompt_text}) # 纯文本

    # --- 调用 API 并处理响应 ---
    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, current_model_id, api_messages)

      if response_for_discord:
          # 更新历史记录 (仅文本)
          history_deque.append({"role": "user", "content": user_prompt_text}) # 只存用户文本
          if response_for_history: history_deque.append({"role": "assistant", "content": response_for_history}) # 只存回复文本
          else: logger.warning(f"No history content returned for {channel_id}.")

          # 发送并处理长消息分割
          if len(response_for_discord) <= 2000: await message.channel.send(response_for_discord)
          else:
              logger.warning(f"Response for {channel_id} too long ({len(response_for_discord)}), splitting.")
              parts = []; current_pos = 0
              while current_pos < len(response_for_discord):
                    cut_off = min(current_pos + 1990, len(response_for_discord)); split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                    if split_index <= current_pos: space_split_index = response_for_discord.rfind(' ', current_pos, cut_off); split_index = space_split_index if space_split_index > current_pos else cut_off
                    chunk_to_check = response_for_discord[current_pos:split_index]
                    if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0: fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1); split_index = fallback_split if fallback_split > current_pos else split_index
                    parts.append(response_for_discord[current_pos:split_index]); current_pos = split_index
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace(): current_pos += 1
              for i, part in enumerate(parts):
                  if not part.strip(): continue
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0: await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  await message.channel.send(part.strip())
      elif response_for_history: # API 返回错误信息
            await message.channel.send(f"抱歉，处理请求时出错：\n{response_for_history}")
      else: # 未知错误
          logger.error(f"Unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时未知问题。")
    except Exception as e: # 捕获 on_message 中的所有其他异常
        logger.exception(f"Error in on_message handler for channel {channel_id}")
        try: await message.channel.send(f"处理消息时内部错误: {e}")
        except Exception: pass

# --- 运行 Bot ---
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN: logger.critical("未设置 DISCORD_BOT_TOKEN"); exit(1)
    if not DEEPSEEK_API_KEY: logger.critical("未设置 DEEPSEEK_API_KEY"); exit(1)
    try: logger.info("尝试启动 Discord 机器人..."); bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure: logger.critical("Token 无效，登录失败。")
    except discord.PrivilegedIntentsRequired as e: logger.critical(f"必需 Intents 未启用！请开启 Message Content Intent。错误: {e}")
    except Exception as e: logger.exception("启动机器人时发生错误。")